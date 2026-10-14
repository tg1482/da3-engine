"""PyTorch DA3 outputs to test the engine against (needs torch + depth_anything_3).

    python tools/export_reference.py depth-anything/DA3-SMALL frames/*.jpg --out ref/da3-small.npz

Runs DA3 itself (DepthAnything3Net, float32, CPU) on the images as one multi-view batch and saves: the raw RGB
frames, DA3's preprocessed images, its interpolated positional table, view 0's tokens into and out of the last
per-view block, the out-layer tokens and camera tokens, depth/confidence and rays/confidence (DualDPT), camera-head
extrinsics/intrinsics, and the ray-solver extrinsics/intrinsics (use_ray_pose, seed 0).
"""
import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # before torch: torch and DA3's deps each bundle OpenMP

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--res", type=int, default=504)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from depth_anything_3.api import DepthAnything3
    api = DepthAnything3.from_pretrained(a.model).eval()
    net = api.model.float()
    vit = net.backbone.pretrained
    rgb = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in a.images])
    imgs, _, _ = api.input_processor(list(rgb), None, None, a.res, "upper_bound_resize")
    x = imgs.float()[None]                                             # 1,S,3,H,W
    H, W = x.shape[-2:]
    out = {"rgb": rgb, "images": x[0].permute(0, 2, 3, 1).numpy()}
    with torch.no_grad():
        out["pos_table"] = vit.interpolate_pos_encoding(torch.zeros(1, 1 + (H // 14) * (W // 14), vit.embed_dim), H, W).numpy()
        t = vit.prepare_tokens_with_masks(x[:, :1])
        for blk in vit.blocks[:vit.alt_start - 1]:
            t = vit.process_attention(t, blk, "local", pos=None)
        out["early_before"] = t[0].numpy()
        out["early_x"] = vit.process_attention(t, vit.blocks[vit.alt_start - 1], "local", pos=None)[0].numpy()
        feats, _ = net.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy="saddle_balanced")
        for i, (tok, cam) in enumerate(feats):
            out[f"feat{i}"], out[f"cam{i}"] = tok[0].numpy(), cam[0].numpy()
        hd = net.head(feats, H, W, patch_start_idx=0)
        out["ray"], out["ray_conf"] = hd["ray"][0].numpy(), hd["ray_conf"][0].numpy()
        r = net(x)
        out["depth"], out["conf"] = r.depth[0].numpy(), r.depth_conf[0].numpy()
        out["w2c"], out["K"] = r.extrinsics[0][:, :3].numpy(), r.intrinsics[0].numpy()
        torch.manual_seed(0)
        r = net(x, use_ray_pose=True)
        out["ray_w2c"], out["ray_K"] = r.extrinsics[0][:, :3].numpy(), r.intrinsics[0].numpy()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)
    print(a.out, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
