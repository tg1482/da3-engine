# da3-engine

Fast multi-view [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) inference for streaming
(SLAM, video). PyTorch or MLX backend, same outputs as DA3.

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

r = eng.infer(views, depth=[5], rays=[5], ray_pose=True)
r.w2c, r.K, r.depth[5], r.depth_conf[5], r.ray[5], r.ray_w2c[5]
```

## Changes vs DA3

All exact (same outputs, fp16 rounding aside):
- Per-view blocks run once per image and are kept (keyframes are never recomputed).
- Heads run only for requested views; the ray branch only when rays are requested.
- Weights cast to fp16 once (no autocast re-casting every call).
- RoPE tables precomputed (DA3 syncs the GPU in every block).
- MLX buffer cache capped (uncapped, LARGE swapped a 16 GB Mac).

No K/V cache across frames: DA3's cross-view attention is bidirectional, so keyframes' keys/values change with
every new frame. Freezing them was 3–4× faster but doubled SLAM trajectory error (translation direction off ~18°).

## Benchmarks

One SLAM step (new image + 5 keyframes, 504×378, fp16), M2 Pro 16 GB, ms:

| model | DA3 (PyTorch, autocast) | da3-engine, torch | da3-engine, mlx |
|---|---|---|---|
| SMALL | 213 | **166** | 168 |
| BASE | 480 | **371** | 387 |
| LARGE-1.1 | 1378 | **1076** | 1111 |

Ray pose (DA3's CPU RANSAC solver) adds ~550 ms. Run: `python tools/bench.py <model> <ref.npz>`.

## Tests

Both backends against DA3's `DepthAnything3Net`: fp32 within ~1e-6 (depth, confidence, rays, poses), fp16 within
1 %. Generate references with `tools/export_reference.py`, then `pytest`.

## License

Apache-2.0, as Depth Anything 3's code. Model weights keep their own licenses (see each model card).
