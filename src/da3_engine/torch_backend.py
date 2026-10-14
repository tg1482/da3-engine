"""The engine's operations on DA3's own PyTorch modules (depth_anything_3), run by our forward pass.

Same arithmetic as DepthAnything3Net; what our pass changes for speed:
  - weights converted to half once (model.half()), instead of autocast re-casting every weight on every call
  - rotary sines/cosines built once per grid; DA3's RoPE calls int(positions.max()) in every block, a GPU->CPU sync
  - the per-view blocks run per image and are kept; heads run only for the views and outputs asked for
  - the ray (aux) branch of DualDPT runs only when rays are asked for
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # before torch: torch and DA3's deps each bundle OpenMP

import numpy as np
import torch
import torch.nn.functional as F

from .common import THRESH_FOR_REF_SELECTION, rope_positions, saddle_balanced, uv_embed


def _interp(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


class TorchBackend:
    name = "torch"

    def __init__(self, model: str, half: bool = True, device: str | None = None):
        from depth_anything_3.api import DepthAnything3
        dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else
                                      "cuda" if torch.cuda.is_available() else "cpu"))
        api = DepthAnything3.from_pretrained(model).to(dev).eval()
        self.net = api.model
        self.vit = self.net.backbone.pretrained
        self.out_layers = list(self.net.backbone.out_layers)
        self.dev = dev
        self.dtype = torch.float16 if half else torch.float32
        self._pe32 = self.vit.pos_embed.detach().float().clone()   # interpolated in float32, as DA3 does
        if half:
            self.vit.half()
            self.net.head.half()                        # the camera head stays float32, as in DA3
            for m in self._aux_modules():
                m.float()
        self._pos, self._rope = {}, {}             # per-shape tables: positions, rotary, uv
        torch.set_grad_enabled(False)

    # ---------------------------------------------------------------------------------------------------- backbone

    def _pos_table(self, ph, pw):
        if (ph, pw) not in self._pos:
            x = torch.zeros(1, 1 + ph * pw, self.vit.embed_dim, device=self.dev)
            kept = self.vit.pos_embed
            self.vit.pos_embed = torch.nn.Parameter(self._pe32, requires_grad=False)
            try:
                pe = self.vit.interpolate_pos_encoding(x, ph * 14, pw * 14)   # DA3 names (H, W) as (w, h)
            finally:
                self.vit.pos_embed = kept
            self._pos[ph, pw] = pe.to(self.dtype)
        return self._pos[ph, pw]

    def _tables(self, pos: np.ndarray):
        key = pos.tobytes()
        if key not in self._rope:
            hd = self.vit.embed_dim // self.vit.num_heads
            d = hd // 2
            inv = 1.0 / (self.vit.rope.base_frequency ** (np.arange(0, d, 2, dtype=np.float32) / d))
            out = []
            for axis in (0, 1):
                ang = pos[:, axis].astype(np.float32)[:, None] * inv[None]
                ang = np.concatenate([ang, ang], -1)
                out += [torch.from_numpy(np.cos(ang)).to(self.dev, self.dtype),
                        torch.from_numpy(np.sin(ang)).to(self.dev, self.dtype)]
            self._rope[key] = out
        return self._rope[key]

    def _tabs(self, i, ph, pw, S):
        if self.vit.rope is None or i < self.vit.rope_start:
            return None, None
        lpos, gpos = rope_positions(ph, pw)
        return self._tables(lpos), self._tables(np.tile(gpos, (S, 1)))

    @staticmethod
    def _rope_apply(t, tabs):
        cy, sy, cx, sx = tabs
        h = t.shape[-1] // 2

        def rot(x):
            q = x.shape[-1] // 2
            return torch.cat([-x[..., q:], x[..., :q]], -1)
        v, w = t[..., :h], t[..., h:]
        return torch.cat([v * cy + rot(v) * sy, w * cx + rot(w) * sx], -1)

    def _block(self, blk, x, tabs=None):
        a = blk.attn
        b, n, c = x.shape
        h = a.num_heads
        qkv = a.qkv(blk.norm1(x)).reshape(b, n, 3, h, c // h).permute(2, 0, 3, 1, 4)
        q, k, v = a.q_norm(qkv[0]), a.k_norm(qkv[1]), qkv[2]
        if tabs is not None:
            q, k = self._rope_apply(q, tabs), self._rope_apply(k, tabs)
        o = F.scaled_dot_product_attention(q, k, v)
        x = x + blk.ls1(a.proj(o.transpose(1, 2).reshape(b, n, c)))
        return x + blk.ls2(blk.mlp(blk.norm2(x)))

    def early(self, img: np.ndarray) -> dict:
        x = torch.from_numpy(img).to(self.dev, self.dtype).permute(2, 0, 1)[None]
        x = self.vit.patch_embed(x)                     # (1, N, C)
        ph, pw = img.shape[0] // 14, img.shape[1] // 14
        x = torch.cat([self.vit.cls_token.to(x.dtype), x], 1) + self._pos_table(ph, pw)
        a = self.vit.alt_start
        for blk in self.vit.blocks[:a - 1]:
            x = self._block(blk, x)
        before = x
        x = self._block(self.vit.blocks[a - 1], x)
        return {"before": before, "x": x, "grid": (ph, pw)}

    def _out(self, local_x, x):
        c = self.vit.embed_dim
        o = torch.cat([local_x, x], -1)
        return torch.cat([o[..., :c], self.vit.norm(o[..., c:])], -1)[:, 1:], o[:, 0]

    def cross(self, views: list):
        vit = self.vit
        S = len(views)
        ph, pw = views[0]["grid"]
        x = torch.cat([v["x"] for v in views], 0)
        order = list(range(S))
        if S >= THRESH_FOR_REF_SELECTION:
            cls = torch.cat([v["before"][:, 0] for v in views], 0).float().cpu().numpy()
            r = saddle_balanced(cls)
            order = [r] + [i for i in range(S) if i != r]
            x = x[order]
        inv = [order.index(j) for j in range(S)]
        local_x = x
        T = x.shape[1]
        outs = []
        for i in range(vit.alt_start, len(vit.blocks)):
            blk = vit.blocks[i]
            ltabs, gtabs = self._tabs(i, ph, pw, S)
            if i == vit.alt_start:
                cam = torch.cat([vit.camera_token[:, :1], vit.camera_token[:, 1:].expand(1, S - 1, -1)], 1)
                x = torch.cat([cam[0][:, None].to(x.dtype), x[:, 1:]], 1)
                local_x = x
            if i % 2 == 1:
                x = self._block(blk, x.reshape(1, S * T, -1), gtabs).reshape(S, T, -1)
            else:
                x = self._block(blk, x, ltabs)
                local_x = x
            if i in self.out_layers:
                tok, cam = self._out(local_x, x)
                outs.append((tok[inv], cam[inv]))
        return outs

    @staticmethod
    def select(outs, idx):
        idx = list(idx)
        return [(t[idx], c[idx]) for t, c in outs]

    # ------------------------------------------------------------------------------------------------------- heads

    def _aux_modules(self):
        """The ray branch, kept in float32: its five unactivated convs overflow half (BASE gave NaN rays)."""
        sc = self.net.head.scratch
        return [sc.refinenet1_aux, sc.refinenet2_aux, sc.refinenet3_aux, sc.refinenet4_aux,
                sc.output_conv1_aux[-1], sc.output_conv2_aux[-1]]

    def _uv(self, x, W: int, H: int):
        """DualDPT's _add_pos_embed, its table built once per shape and kept in the activations' dtype (DA3's is
        float32 always, which would promote a half head back to float32)."""
        key = tuple(x.shape[1:]) + (W, H, x.dtype)
        if key not in self._rope:
            pe = uv_embed(x.shape[3], x.shape[2], W / H, x.shape[1])
            self._rope[key] = torch.from_numpy(pe).permute(2, 0, 1)[None].to(self.dev, x.dtype)
        return x + self._rope[key]

    def heads(self, outs, H: int, W: int, depth: bool, rays: bool) -> dict:
        hd = self.net.head
        sc = hd.scratch
        ph, pw = H // 14, W // 14
        maps = []
        for i, (t, _) in enumerate(outs):
            s, n, c = t.shape
            x = hd.norm(t.to(self.dtype)).permute(0, 2, 1).reshape(s, c, ph, pw)
            x = self._uv(hd.projects[i](x), W, H)
            maps.append(hd.resize_layers[i](x))
        l1, l2, l3, l4 = sc.layer1_rn(maps[0]), sc.layer2_rn(maps[1]), sc.layer3_rn(maps[2]), sc.layer4_rn(maps[3])
        out = {}
        if depth:
            y = sc.refinenet4(l4, size=l3.shape[2:])
            y = sc.refinenet3(y, l3, size=l2.shape[2:])
            y = sc.refinenet2(y, l2, size=l1.shape[2:])
            y = sc.output_conv1(sc.refinenet1(y, l1))
            y = self._uv(_interp(y, (ph * 14, pw * 14)), W, H)
            logits = sc.output_conv2(y).float()        # exp in float32: no fp16 overflow
            out["depth"], out["depth_conf"] = torch.exp(logits[:, 0]), torch.exp(logits[:, 1]) + 1.0
        if rays:
            l1, l2, l3, l4 = l1.float(), l2.float(), l3.float(), l4.float()
            y = sc.refinenet4_aux(l4, size=l3.shape[2:])
            y = sc.refinenet3_aux(y, l3, size=l2.shape[2:])
            y = sc.refinenet2_aux(y, l2, size=l1.shape[2:])
            y = sc.output_conv1_aux[-1](sc.refinenet1_aux(y, l1))
            logits = sc.output_conv2_aux[-1](self._uv(y, W, H)).float().permute(0, 2, 3, 1)
            out["ray"], out["ray_conf"] = logits[..., :6], torch.exp(logits[..., 6]) + 1.0
        return {k: v.cpu().numpy() for k, v in out.items()}

    def pose_enc(self, outs) -> np.ndarray:
        return self.net.cam_dec(outs[-1][1].float()[None])[0].float().cpu().numpy().astype(np.float64)
