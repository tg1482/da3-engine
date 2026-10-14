"""The DA3 engine: multi-view Depth Anything 3 for streaming use, on PyTorch or MLX.

    eng = Engine("depth-anything/DA3-SMALL", backend="mlx")      # or backend="torch"
    v = eng.view(rgb)                                             # once per image: preprocess + per-view blocks

    # exact (the same outputs as DA3 for these views)
    r = eng.infer([v0, v1, ..., v5], depth=[5], rays=[], ray_pose=False)

    # streaming with a K/V cache: run a set of views once and keep their keys/values; each new view then attends
    # to them instead of re-running all of them (they don't see the new view: an approximation, measured in tests)
    ctx, r = eng.context([v0, ..., v4], depth=[4])
    r = eng.extend(ctx, v5, depth=True, rays=True, ray_pose=True)

Result: w2c (S x 3 x 4) and K (S x 3 x 3) from the camera head for every view in the step (for extend: the context's
views then the new one, all in the context's frame); per requested view: depth, depth_conf (H x W), ray (h x w x 6:
direction, origin), ray_conf; ray_w2c / ray_K from the rays when ray_pose (DA3's use_ray_pose solver).
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


@dataclass(frozen=True)
class Context:
    """A K/V cache: views run together once, with every cross-view block's keys and values kept."""
    kvs: dict
    enc: np.ndarray                 # camera-head pose encodings of the context's views (S x 9)
    hw: tuple
    size: int


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
        return self._run(views, depth, rays, ray_pose, keep_kv=False)[1]

    def context(self, views: list, depth=(), rays=(), ray_pose: bool = False):
        """infer, keeping the cross-view keys/values: (Context, Result)."""
        return self._run(views, depth, rays, ray_pose, keep_kv=True)

    def _run(self, views, depth, rays, ray_pose, keep_kv):
        H, W = views[0].hw
        outs, kvs = self.b.cross([v.tokens for v in views], keep_kv)
        enc = self.b.pose_enc(outs)
        w2c, K = pose_to_w2c_K(enc, H, W)
        r = Result(w2c, K)
        self._heads(r, outs, list(range(len(views))), H, W, set(depth), set(rays), ray_pose)
        return (Context(kvs, enc, (H, W), len(views)) if keep_kv else None), r

    def extend(self, ctx: Context, view: View, depth: bool = True, rays: bool = False,
               ray_pose: bool = False) -> Result:
        """One new view against a context: its outputs, and every camera in the context's frame (new view last)."""
        H, W = ctx.hw
        assert view.hw == ctx.hw, "a context and its new views share one input size"
        outs = self.b.extend(ctx.kvs, view.tokens)
        enc = np.concatenate([ctx.enc, self.b.pose_enc(outs)], 0)
        w2c, K = pose_to_w2c_K(enc, H, W)
        r = Result(w2c, K)
        n = ctx.size
        self._heads(r, outs, [n], H, W, {n} if depth else set(), {n} if rays else set(), ray_pose)
        return r
