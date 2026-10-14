"""One SLAM step's latency per backend, fp16, on the reference frames (6 views).

    python tools/bench.py depth-anything/DA3-SMALL ref/da3-small.npz --backend mlx torch

step: the new image's per-view blocks + the cross-view blocks over 5 kept views and it, depth for 2 views.
+rays: the same with rays and the ray-solver pose for the new view.
"""
import argparse
import time

import numpy as np

from da3_engine import Engine


def timed(f, n=10):
    for _ in range(3):
        f()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return 1000 * np.median(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ref")
    ap.add_argument("--backend", nargs="+", default=["mlx", "torch"])
    a = ap.parse_args()
    rgb = np.load(a.ref)["rgb"]
    for b in a.backend:
        eng = Engine(a.model, backend=b)
        kept = [eng.view(im) for im in rgb[:-1]]
        new = rgb[-1]
        rows = {
            "view (per image)": lambda: eng.view(new),
            "step": lambda: eng.infer(kept + [eng.view(new)], depth=[4, 5]),
            "step +rays+ray pose": lambda: eng.infer(kept + [eng.view(new)], depth=[4, 5], rays=[5], ray_pose=True),
        }
        for k, f in rows.items():
            print(f"{b} {a.model.split('/')[-1]} {k}: {timed(f):.0f} ms", flush=True)
        del eng, kept


if __name__ == "__main__":
    main()
