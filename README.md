# da3-engine

Fast multi-view [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) inference for streaming
(SLAM, video). PyTorch or MLX backend. Exact by default; optional K/V cache for streaming.

## Install

```sh
pip install -e .                         # MLX backend
pip install torch "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3"
                                         # PyTorch backend; also needed once to convert weights for MLX
```

## Use

```python
from da3_engine import Engine

eng = Engine("depth-anything/DA3-SMALL", backend="mlx")   # or "torch"
views = [eng.view(rgb) for rgb in frames]                  # once per image

r = eng.infer(views, depth=[5], rays=[5], ray_pose=True)   # exact
r.w2c, r.K, r.depth[5], r.depth_conf[5], r.ray[5], r.ray_w2c[5]

ctx, _ = eng.context(views[:5])                            # streaming: keep the keyframes' K/V
r = eng.extend(ctx, views[5])                              # place a new frame against them
```

## Changes vs DA3

Exact (same outputs, fp16 rounding aside):
- Per-view blocks run once per image and are kept (keyframes are never recomputed).
- Heads run only for requested views; the ray branch only when rays are requested.
- Weights cast to fp16 once (no autocast re-casting every call).
- RoPE tables precomputed (DA3 syncs the GPU in every block).
- MLX buffer cache capped (uncapped, LARGE swapped a 16 GB Mac).

Approximate (opt-in):
- **K/V cache**: `context` keeps every cross-view block's keys/values; `extend` runs only the new frame,
  attending to them. The context doesn't see the new frame. On test frames the error vs exact (0.08–0.11 pose,
  3–7 % depth) is about what exact DA3 itself changes by when one view is added to the set (0.09–0.27, 8–15 %).

## Benchmarks

One SLAM step (new image + 5 keyframes, 504×378, fp16), M2 Pro 16 GB, ms:

| model | DA3 (PyTorch, autocast) | exact, torch | exact, mlx | K/V, torch | K/V, mlx |
|---|---|---|---|---|---|
| SMALL | 213 | 166 | 168 | **45** | 50 |
| BASE | 480 | 371 | 387 | **97** | 109 |
| LARGE-1.1 | 1378 | 1076 | 1111 | **269** | 297 |

Ray pose (DA3's CPU RANSAC solver) adds ~550 ms. Run: `python tools/bench.py <model> <ref.npz>`.

## Tests

Both backends against DA3's `DepthAnything3Net`: fp32 within ~1e-6 (depth, confidence, rays, poses), fp16 within
1 %. Generate references with `tools/export_reference.py`, then `pytest`.

## License

Apache-2.0, as Depth Anything 3's code. Model weights keep their own licenses (see each model card).
