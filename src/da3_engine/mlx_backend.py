"""Depth Anything 3 in MLX: the network, and the backend operations the engine composes.

A port of ByteDance's depth_anything_3 PyTorch model (DepthAnything3Net: DinoV2 backbone, DualDPT head, CameraDec):
  backbone  DinoV2 ViT (S/B/L): patch embedding, class token, positional embedding interpolated for the input size
            (baked by convert.py with PyTorch's own bicubic interpolation), blocks with LayerScale, query/key
            LayerNorm from `qknorm_start`, 2D rotary position embedding from `rope_start`; the first `alt_start`
            blocks see each view alone, after that odd blocks attend across all views and even ones within a view;
            camera tokens replace the class token at `alt_start`; the reference view (saddle_balanced) goes first
            for the cross-view blocks and the outputs are put back in order
  head      DualDPT: the main branch (depth + confidence) and, on request, the aux branch (ray maps + confidence)
  camera    CameraDec: translation, quaternion and field of view per view
Differences from PyTorch, none of which change the arithmetic: rotary tables built once per grid; the per-view
blocks run separately per image (`early`) so their output can be kept; the depth head runs only for the views asked.

Layouts: images NHWC; tokens (S, T, C) per view or (1, S*T, C) across views.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .common import THRESH_FOR_REF_SELECTION, pos_table, rope_positions, saddle_balanced, uv_embed


class Memo:
    """A cache an nn.Module can hold without MLX taking its arrays for parameters."""

    def __init__(self):
        self.d = {}


@dataclass
class Config:
    dim: int
    depth: int
    heads: int
    alt_start: int
    rope_start: int
    qknorm_start: int
    out_layers: list
    head_features: int
    head_out_channels: list
    qkv_bias: bool = True
    patch: int = 14
    rope_freq: float = 100.0

    @staticmethod
    def load(path: str) -> "Config":
        return Config(**json.load(open(path)))


# --------------------------------------------------------------------------------------------------------------------
# backbone
# --------------------------------------------------------------------------------------------------------------------

class Rope2D:
    """DA3's RotaryPositionEmbedding2D, with its tables built once per positions and reused by every block."""

    def __init__(self, freq: float):
        self.freq = freq
        self._cache: dict = {}

    def tables(self, pos: np.ndarray, head_dim: int):
        """pos (N, 2) int y, x -> (cos_y, sin_y, cos_x, sin_x), each (1, 1, N, head_dim/2)."""
        key = (pos.tobytes(), head_dim)
        if key not in self._cache:
            d = head_dim // 2
            inv = 1.0 / (self.freq ** (np.arange(0, d, 2, dtype=np.float32) / d))
            out = []
            for axis in (0, 1):
                ang = pos[:, axis].astype(np.float32)[:, None] * inv[None, :]
                ang = np.concatenate([ang, ang], axis=-1)
                out += [mx.array(np.cos(ang))[None, None], mx.array(np.sin(ang))[None, None]]
            self._cache[key] = tuple(out)
        return self._cache[key]

    @staticmethod
    def _rot(x):
        h = x.shape[-1] // 2
        return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)

    def __call__(self, t, tabs):
        cy, sy, cx, sx = (a.astype(t.dtype) for a in tabs)
        h = t.shape[-1] // 2
        v, w = t[..., :h], t[..., h:]
        return mx.concatenate([v * cy + self._rot(v) * sy, w * cx + self._rot(w) * sx], axis=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, qkv_bias: bool, qk_norm: bool):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(dim // heads, eps=1e-5)
            self.k_norm = nn.LayerNorm(dim // heads, eps=1e-5)

    def __call__(self, x, rope=None, tabs=None):
        b, n, c = x.shape
        h = self.heads
        qkv = self.qkv(x).reshape(b, n, 3, h, c // h).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = rope(q, tabs), rope(k, tabs)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=(c // h) ** -0.5)
        return self.proj(o.transpose(0, 2, 1, 3).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, qkv_bias: bool, qk_norm: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads, qkv_bias, qk_norm)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)
        self.ls1 = mx.ones((dim,))
        self.ls2 = mx.ones((dim,))

    def __call__(self, x, rope=None, tabs=None):
        x = x + self.ls1 * self.attn(self.norm1(x), rope, tabs)
        return x + self.ls2 * self.mlp(self.norm2(x))


class Backbone(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.patch_embed = nn.Conv2d(3, d, kernel_size=cfg.patch, stride=cfg.patch)
        self.cls_token = mx.zeros((1, 1, d))
        self.camera_token = mx.zeros((1, 2, d))
        self.blocks = [Block(d, cfg.heads, cfg.qkv_bias, cfg.qknorm_start != -1 and i >= cfg.qknorm_start)
                       for i in range(cfg.depth)]
        self.norm = nn.LayerNorm(d, eps=1e-5)
        self.rope = Rope2D(cfg.rope_freq) if cfg.rope_start != -1 else None
        self.pos_embed = mx.zeros((1, 1 + 37 * 37, d))   # DinoV2's, for its 518 px grid; resampled per input size
        self._pos = Memo()

    def pos_tables(self, ph: int, pw: int):
        d = self._pos.d
        if (ph, pw) not in d:
            d[ph, pw] = mx.array(pos_table(np.array(self.pos_embed.astype(mx.float32)), ph, pw)).astype(
                self.patch_embed.weight.dtype)
        return d[ph, pw]

    def early(self, img):
        """One image (1, H, W, 3) through the per-view blocks: the tokens into block alt_start-1 (the reference view
        is chosen by them) and out of it."""
        x = self.patch_embed(img)
        _, ph, pw, c = x.shape
        x = x.reshape(1, ph * pw, c)
        x = mx.concatenate([self.cls_token.astype(x.dtype), x], axis=1) + self.pos_tables(ph, pw)
        for blk in self.blocks[:self.cfg.alt_start - 1]:
            x = blk(x)
        before = x
        x = self.blocks[self.cfg.alt_start - 1](x)
        return {"before": before, "x": x, "grid": (ph, pw)}

    def _tabs(self, i, ph, pw, S):
        """Rotary tables for block i: (local, global) or (None, None) before rope_start."""
        cfg = self.cfg
        if self.rope is None or i < cfg.rope_start:
            return None, None
        lpos, gpos = rope_positions(ph, pw)
        hd = cfg.dim // cfg.heads
        return self.rope.tables(lpos, hd), self.rope.tables(np.tile(gpos, (S, 1)), hd)

    def _out(self, local_x, x):
        c = self.cfg.dim
        o = mx.concatenate([local_x, x], axis=-1)
        return mx.concatenate([o[..., :c], self.norm(o[..., c:])], axis=-1)[:, 1:], o[:, 0]

    def cross(self, views: list):
        """The cross-view blocks over every view's early tokens: [(tokens S x N x 2C, camera S x 2C)] per out layer,
        in the views' own order."""
        cfg = self.cfg
        S = len(views)
        ph, pw = views[0]["grid"]
        x = mx.concatenate([v["x"] for v in views], axis=0)               # (S, T, C)
        order = list(range(S))
        if S >= THRESH_FOR_REF_SELECTION:
            r = saddle_balanced(np.array(mx.concatenate([v["before"][:, 0] for v in views], 0).astype(mx.float32)))
            order = [r] + [i for i in range(S) if i != r]
            x = x[mx.array(order)]
        inv = mx.array([order.index(j) for j in range(S)])
        local_x = x
        T = x.shape[1]
        outs = []
        for i in range(cfg.alt_start, cfg.depth):
            blk = self.blocks[i]
            ltabs, gtabs = self._tabs(i, ph, pw, S)
            rope = self.rope if ltabs is not None else None
            if i == cfg.alt_start:
                cam = mx.concatenate([self.camera_token[:, :1],
                                      mx.broadcast_to(self.camera_token[:, 1:], (1, S - 1, cfg.dim))], axis=1)
                x = mx.concatenate([cam[0][:, None].astype(x.dtype), x[:, 1:]], axis=1)
                local_x = x                             # DA3 writes the camera token in place, into local_x too
            if i % 2 == 1:
                x = blk(x.reshape(1, S * T, -1), rope, gtabs).reshape(S, T, -1)
            else:
                x = blk(x, rope, ltabs)
                local_x = x
            if i in cfg.out_layers:
                tok, cam = self._out(local_x, x)
                outs.append((tok[inv], cam[inv]))
        return outs

# --------------------------------------------------------------------------------------------------------------------
# head (DualDPT)
# --------------------------------------------------------------------------------------------------------------------

def bilinear_resize(x, out_h: int, out_w: int):
    """align_corners=True bilinear (NHWC)."""
    b, h, w, c = x.shape
    if (h, w) == (out_h, out_w):
        return x
    ys = mx.arange(out_h, dtype=mx.float32) * ((h - 1) / max(out_h - 1, 1))
    xs = mx.arange(out_w, dtype=mx.float32) * ((w - 1) / max(out_w - 1, 1))
    y0, x0 = mx.floor(ys).astype(mx.int32), mx.floor(xs).astype(mx.int32)
    y1, x1 = mx.minimum(y0 + 1, h - 1), mx.minimum(x0 + 1, w - 1)
    wy = (ys - y0.astype(mx.float32))[None, :, None, None].astype(x.dtype)
    wx = (xs - x0.astype(mx.float32))[None, None, :, None].astype(x.dtype)
    top, bot = mx.take(x, y0, axis=1), mx.take(x, y1, axis=1)
    tl, tr = mx.take(top, x0, axis=2), mx.take(top, x1, axis=2)
    bl, br = mx.take(bot, x0, axis=2), mx.take(bot, x1, axis=2)
    return tl * (1 - wy) * (1 - wx) + tr * (1 - wy) * wx + bl * wy * (1 - wx) + br * wy * wx


class RCU(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.conv1 = nn.Conv2d(f, f, 3, padding=1)
        self.conv2 = nn.Conv2d(f, f, 3, padding=1)

    def __call__(self, x):
        return self.conv2(nn.relu(self.conv1(nn.relu(x)))) + x


class Fusion(nn.Module):
    def __init__(self, f, residual=True):
        super().__init__()
        self.residual = residual
        if residual:
            self.resConfUnit1 = RCU(f)
        self.resConfUnit2 = RCU(f)
        self.out_conv = nn.Conv2d(f, f, 1)

    def __call__(self, y, lat=None, size=None):
        if self.residual and lat is not None:
            y = y + self.resConfUnit1(lat)
        y = self.resConfUnit2(y)
        size = size or (y.shape[1] * 2, y.shape[2] * 2)
        return self.out_conv(bilinear_resize(y, *size))


class Head(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        din, f, oc = 2 * cfg.dim, cfg.head_features, cfg.head_out_channels
        self.patch = cfg.patch
        self.norm = nn.LayerNorm(din, eps=1e-5)
        self.projects = [nn.Conv2d(din, c, 1) for c in oc]
        self.resize0 = nn.ConvTranspose2d(oc[0], oc[0], 4, stride=4)
        self.resize1 = nn.ConvTranspose2d(oc[1], oc[1], 2, stride=2)
        self.resize3 = nn.Conv2d(oc[3], oc[3], 3, stride=2, padding=1)
        self.layer_rn = [nn.Conv2d(c, f, 3, padding=1, bias=False) for c in oc]
        self.refinenet1, self.refinenet2, self.refinenet3 = Fusion(f), Fusion(f), Fusion(f)
        self.refinenet4 = Fusion(f, residual=False)
        self.output_conv1 = nn.Conv2d(f, f // 2, 3, padding=1)
        self.output_conv2_0 = nn.Conv2d(f // 2, 32, 3, padding=1)
        self.output_conv2_2 = nn.Conv2d(32, 2, 1)
        # aux (ray) branch: its own fusion chain; only the finest level reaches the output, as in DualDPT
        self.refinenet1_aux, self.refinenet2_aux, self.refinenet3_aux = Fusion(f), Fusion(f), Fusion(f)
        self.refinenet4_aux = Fusion(f, residual=False)
        self.aux_conv1 = [nn.Conv2d(f if k % 2 == 0 else f // 2, f // 2 if k % 2 == 0 else f, 3, padding=1)
                          for k in range(5)]
        self.aux_conv2_0 = nn.Conv2d(f // 2, 32, 3, padding=1)
        self.aux_ln = nn.LayerNorm(32, eps=1e-5)
        self.aux_conv2_5 = nn.Conv2d(32, 7, 1)
        self._pe = Memo()

    def aux_modules(self):
        """The ray branch, kept in float32: its five unactivated convs overflow half (BASE gave NaN rays)."""
        return [self.refinenet1_aux, self.refinenet2_aux, self.refinenet3_aux, self.refinenet4_aux,
                *self.aux_conv1, self.aux_conv2_0, self.aux_ln, self.aux_conv2_5]

    def _pos(self, x, W: int, H: int):
        key = (x.shape[1], x.shape[2], x.shape[3], W, H)
        if key not in self._pe.d:
            self._pe.d[key] = mx.array(uv_embed(x.shape[2], x.shape[1], W / H, x.shape[3]))[None]
        return x + self._pe.d[key].astype(x.dtype)

    def __call__(self, feats, H: int, W: int, depth: bool = True, rays: bool = False):
        """feats: 4 x (S, N, 2C) -> {depth, depth_conf (S, H, W); ray (S, h, w, 6), ray_conf (S, h, w)}."""
        ph, pw = H // self.patch, W // self.patch
        maps = []
        for i, f in enumerate(feats):
            s, n, c = f.shape
            x = self._pos(self.projects[i](self.norm(f).reshape(s, ph, pw, c)), W, H)
            maps.append((self.resize0, self.resize1, lambda t: t, self.resize3)[i](x))
        l1, l2, l3, l4 = [rn(m) for rn, m in zip(self.layer_rn, maps)]
        out = {}
        if depth:
            y = self.refinenet4(l4, size=(l3.shape[1], l3.shape[2]))
            y = self.refinenet3(y, l3, size=(l2.shape[1], l2.shape[2]))
            y = self.refinenet2(y, l2, size=(l1.shape[1], l1.shape[2]))
            y = self.output_conv1(self.refinenet1(y, l1))
            y = self._pos(bilinear_resize(y, ph * self.patch, pw * self.patch), W, H)
            logits = self.output_conv2_2(nn.relu(self.output_conv2_0(y))).astype(mx.float32)   # exp: no fp16 overflow
            out["depth"], out["depth_conf"] = mx.exp(logits[..., 0]), mx.exp(logits[..., 1]) + 1.0
        if rays:
            l1, l2, l3, l4 = (t.astype(mx.float32) for t in (l1, l2, l3, l4))
            y = self.refinenet4_aux(l4, size=(l3.shape[1], l3.shape[2]))
            y = self.refinenet3_aux(y, l3, size=(l2.shape[1], l2.shape[2]))
            y = self.refinenet2_aux(y, l2, size=(l1.shape[1], l1.shape[2]))
            y = self.refinenet1_aux(y, l1)
            for conv in self.aux_conv1:
                y = conv(y)
            y = self._pos(y, W, H)
            logits = self.aux_conv2_5(nn.relu(self.aux_ln(self.aux_conv2_0(y)))).astype(mx.float32)
            out["ray"], out["ray_conf"] = logits[..., :6], mx.exp(logits[..., 6]) + 1.0
        return out


class CameraDec(nn.Module):
    def __init__(self, din: int):
        super().__init__()
        self.backbone_0 = nn.Linear(din, din)
        self.backbone_2 = nn.Linear(din, din)
        self.fc_t = nn.Linear(din, 3)
        self.fc_qvec = nn.Linear(din, 4)
        self.fc_fov_0 = nn.Linear(din, 2)

    def __call__(self, cam):
        f = nn.relu(self.backbone_2(nn.relu(self.backbone_0(cam))))
        return mx.concatenate([self.fc_t(f), self.fc_qvec(f), nn.relu(self.fc_fov_0(f))], axis=-1)


class DA3(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg)
        self.head = Head(cfg)
        self.cam_dec = CameraDec(2 * cfg.dim)


# --------------------------------------------------------------------------------------------------------------------
# backend
# --------------------------------------------------------------------------------------------------------------------

class MLXBackend:
    """The engine's operations on the MLX network. Tokens stay on the GPU; numpy crosses at the edges only."""

    name = "mlx"

    def __init__(self, weights_dir: str, half: bool = True):
        cfg = Config.load(f"{weights_dir}/config.json")
        m = DA3(cfg)
        w = mx.load(f"{weights_dir}/weights.safetensors")
        w["backbone.pos_embed"] = w.pop("pos_embed")
        m.load_weights(list(w.items()), strict=True)
        self.dtype = mx.float16 if half else mx.float32
        if half:
            # backbone and depth head in half (the head is 2x faster and as accurate as in float32: tests); the
            # camera head, which is tiny, stays float32 as in DA3
            m.backbone.set_dtype(self.dtype)
            m.head.set_dtype(self.dtype)
            for a in m.head.aux_modules():
                a.set_dtype(mx.float32)
        m.backbone.pos_embed = w["backbone.pos_embed"]   # float32: resampled in float32, as DA3 does
        mx.eval(m.parameters())
        # MLX keeps freed buffers for reuse; uncapped that grew to 7.5 GB for LARGE and pushed a 16 GB machine into
        # swap (4 s per step instead of 1.2 s)
        mx.set_cache_limit(512 << 20)
        self.m, self.cfg = m, cfg

    def early(self, img: np.ndarray) -> dict:
        e = self.m.backbone.early(mx.array(img[None]).astype(self.dtype))
        mx.eval(e["before"], e["x"])
        return e

    def cross(self, views: list):
        outs = self.m.backbone.cross(views)
        mx.eval(outs)
        return outs

    @staticmethod
    def select(outs, idx):
        sel = mx.array(list(idx))
        return [(t[sel], c[sel]) for t, c in outs]

    def heads(self, outs, H: int, W: int, depth: bool, rays: bool) -> dict:
        r = self.m.head([t.astype(self.dtype) for t, _ in outs], H, W, depth=depth, rays=rays)
        mx.eval(r)
        return {k: np.array(v) for k, v in r.items()}

    def pose_enc(self, outs) -> np.ndarray:
        enc = self.m.cam_dec(outs[-1][1].astype(mx.float32))
        return np.array(enc, dtype=np.float64)
