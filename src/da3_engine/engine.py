"""The DA3 engine: multi-view Depth Anything 3 for streaming use, on PyTorch or MLX.

    eng = Engine("depth-anything/DA3-SMALL", backend="mlx")      # or backend="torch"
    v = eng.view(rgb)                                             # once per image: preprocess + per-view blocks
    r = eng.infer([v0, v1, ..., v5], depth=[5], rays=[], ray_pose=False)   # DA3's outputs for these views

Result: w2c (S x 3 x 4) and K (S x 3 x 3) from the camera head for every view; per requested view: depth,
depth_conf (H x W), ray (h x w x 6: direction, origin), ray_conf; ray_w2c / ray_K from the rays when ray_pose (DA3's
use_ray_pose solver).

No K/V cache across steps: DA3's cross-view attention is bidirectional, so a kept view's keys and values change
whenever another view joins. Freezing them (tried 10-14) kept rotation but put translation direction off ~18 deg and
doubled SLAM error; reusing them exactly would need a causally trained model.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .common import pose_to_w2c_K, prep, ray_pose as solve_ray_pose


@dataclass(frozen=True)
class View:
    tokens: dict                    # the backend's per-view tokens: before/after the last per-view block, grid
    hw: tuple                       # (H, W) as the network sees it


@dataclass
class Result:
    w2c: np.ndarray
    K: np.ndarray
    depth: dict = field(default_factory=dict)
    depth_conf: dict = field(default_factory=dict)
    ray: dict = field(default_factory=dict)
    ray_conf: dict = field(default_factory=dict)
    ray_w2c: dict = field(default_factory=dict)
    ray_K: dict = field(default_factory=dict)


def default_weights_dir(model: str) -> str:
    return os.path.join(os.path.expanduser("~/.cache/da3-engine"), model.replace("/", "--"))


class Engine:
    def __init__(self, model: str = "depth-anything/DA3-SMALL", backend: str = "mlx", res: int = 504,
                 half: bool = True, weights_dir: str | None = None):
        self.model, self.res = model, res
        if backend == "mlx":
            from .mlx_backend import MLXBackend
            d = weights_dir or default_weights_dir(model)
            if not os.path.exists(os.path.join(d, "weights.safetensors")):
                from .convert import convert             # needs torch + depth_anything_3, once per model
                convert(model, d)
            self.b = MLXBackend(d, half=half)
        elif backend == "torch":
            from .torch_backend import TorchBackend
            self.b = TorchBackend(model, half=half)
        else:
            raise ValueError(f"backend must be 'mlx' or 'torch', not {backend!r}")
        self.backend = backend

    def view(self, image: np.ndarray) -> View:
        """RGB uint8 (h, w, 3) -> a View: DA3's preprocessing and the per-view blocks, run once per image."""
        img = prep(image, self.res)
        return View(self.b.early(img), img.shape[:2])

    def _heads(self, r: Result, outs, idx: list, H: int, W: int, depth, rays, ray_pose: bool):
        """Depth for views `depth`, rays for views `rays` (outs indexed like idx)."""
        want = sorted(set(depth) | set(rays))
        if not want:
            return
        pos = [idx.index(i) for i in want]
        h = self.b.heads(self.b.select(outs, pos), H, W, depth=bool(depth), rays=bool(rays))
        for j, i in enumerate(want):
            if i in depth:
                r.depth[i], r.depth_conf[i] = h["depth"][j], h["depth_conf"][j]
            if i in rays:
                r.ray[i], r.ray_conf[i] = h["ray"][j], h["ray_conf"][j]
        if ray_pose and rays:
            ids = sorted(rays)
            w2c, K = solve_ray_pose(np.stack([r.ray[i] for i in ids]), np.stack([r.ray_conf[i] for i in ids]), H, W)
            for j, i in enumerate(ids):
                r.ray_w2c[i], r.ray_K[i] = w2c[j], K[j]

    def infer(self, views: list, depth=(), rays=(), ray_pose: bool = False) -> Result:
        H, W = views[0].hw
        outs = self.b.cross([v.tokens for v in views])
        w2c, K = pose_to_w2c_K(self.b.pose_enc(outs), H, W)
        r = Result(w2c, K)
        self._heads(r, outs, list(range(len(views))), H, W, set(depth), set(rays), ray_pose)
        return r
