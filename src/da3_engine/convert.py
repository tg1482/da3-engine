"""DA3 PyTorch checkpoint -> MLX weights (needs torch + depth_anything_3; run once per model).

    python -m da3_engine.convert depth-anything/DA3-SMALL ~/.cache/da3-engine/depth-anything--DA3-SMALL

Writes config.json and weights.safetensors with MLX layouts (convs O,kh,kw,I; transposed convs too). Kept: the
backbone, both DualDPT branches (the aux one only at the finest level, the only one DualDPT outputs), the camera
head. Left out: the camera encoder (pose conditioning) and the aux branch's unused coarser levels.
"""
import argparse
import json
import os
import re

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # before torch: torch and DA3's deps each bundle OpenMP

import numpy as np

NETS = {384: (12, 6), 768: (12, 12), 1024: (24, 16), 1536: (40, 24)}


def rename(k: str):
    k = k.removeprefix("model.")
    if k.startswith("cam_enc."):
        return None
    m = re.match(r"head\.scratch\.output_conv(\d)_aux\.(\d)\.(\d)\.(.*)", k)
    if m:
        conv, level, idx, rest = m.groups()
        if level != "3":
            return None
        if conv == "1":
            return f"head.aux_conv1.{idx}.{rest}"
        return f"head.{ {'0': 'aux_conv2_0', '2': 'aux_ln', '5': 'aux_conv2_5'}[idx] }.{rest}"
    k = k.replace("backbone.pretrained.", "backbone.").replace("patch_embed.proj.", "patch_embed.")
    k = re.sub(r"\.ls(\d)\.gamma$", r".ls\1", k)
    k = re.sub(r"head\.resize_layers\.(\d)\.", r"head.resize\1.", k)
    k = re.sub(r"head\.scratch\.layer(\d)_rn\.", lambda m: f"head.layer_rn.{int(m.group(1)) - 1}.", k)
    k = k.replace("head.scratch.", "head.")
    k = re.sub(r"output_conv2\.(\d)\.", r"output_conv2_\1.", k)
    k = re.sub(r"cam_dec\.backbone\.(\d)\.", r"cam_dec.backbone_\1.", k)
    return k.replace("cam_dec.fc_fov.0.", "cam_dec.fc_fov_0.")


def convert(model: str, out_dir: str):
    import torch
    from safetensors.numpy import save_file
    from depth_anything_3.api import DepthAnything3
    net = DepthAnything3.from_pretrained(model).eval().model
    vit = net.backbone.pretrained
    out = {}
    for k, v in net.state_dict().items():
        n = "pos_embed" if k.endswith("pretrained.pos_embed") else rename(k)
        if n is None:
            continue
        v = v.float().numpy()
        if v.ndim == 4:
            # torch Conv2d (O,I,kh,kw) and ConvTranspose2d (I,O,kh,kw) -> MLX (O,kh,kw,I)
            v = v.transpose(1, 2, 3, 0) if re.search(r"resize[01]\.", n) else v.transpose(0, 2, 3, 1)
        out[n] = np.ascontiguousarray(v)
    depth, heads = NETS[vit.embed_dim]
    cfg = {"dim": vit.embed_dim, "depth": depth, "heads": heads, "alt_start": vit.alt_start,
           "rope_start": vit.rope_start, "qknorm_start": vit.qknorm_start,
           "out_layers": list(net.backbone.out_layers),
           "head_features": net.head.scratch.layer1_rn.out_channels,
           "head_out_channels": [p.out_channels for p in net.head.projects]}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    save_file(out, os.path.join(out_dir, "weights.safetensors"))
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    a = ap.parse_args()
    print(convert(a.model, a.out))
