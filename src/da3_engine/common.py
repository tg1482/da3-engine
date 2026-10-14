"""What both backends share, in numpy: DA3's preprocessing, reference-view choice, positions, pose decoding."""
from __future__ import annotations

import cv2
import numpy as np

PATCH = 14
THRESH_FOR_REF_SELECTION = 3          # DA3: views are reordered around a reference view from 3 views up
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def prep(image: np.ndarray, res: int) -> np.ndarray:
    """RGB uint8 (h, w, 3) -> (H, W, 3) float32 as DA3's InputProcessor makes it ("upper_bound_resize"): longest
    side to `res`, each side to the nearest multiple of 14, ImageNet normalisation. Same cv2 calls as DA3."""
    img = np.ascontiguousarray(image[..., :3])
    h, w = img.shape[:2]
    if max(w, h) != res:
        s = res / float(max(w, h))
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA)
        h, w = nh, nw

    def nearest(x):
        down = (x // PATCH) * PATCH
        return down + PATCH if abs(down + PATCH - x) <= abs(x - down) else down

    nw, nh = max(1, nearest(w)), max(1, nearest(h))
    if (nw, nh) != (w, h):
        up = nw > w or nh > h
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC if up else cv2.INTER_AREA)
    return ((img.astype(np.float32) / 255.0 - MEAN) / STD).astype(np.float32)


def saddle_balanced(cls_tokens: np.ndarray) -> int:
    """DA3's reference view: the one most middle-of-the-road in similarity, norm and spread of its class token."""
    S = len(cls_tokens)
    f = cls_tokens / np.linalg.norm(cls_tokens, axis=-1, keepdims=True)
    score = (f @ f.T - np.eye(S)).sum(-1) / (S - 1)
    norm = np.linalg.norm(cls_tokens, axis=-1)
    var = f.var(-1, ddof=1)

    def nm(m):
        return (m - m.min()) / (m.max() - m.min() + 1e-8)
    return int(np.argmin(np.abs(nm(score) - 0.5) + np.abs(nm(norm) - 0.5) + np.abs(nm(var) - 0.5)))


def rope_positions(ph: int, pw: int):
    """DA3's rotary positions for one view: per-view blocks use the patch grid (y+1, x+1), cross-view blocks all 1
    (no relative position across views); the class/camera token is 0."""
    yy, xx = np.meshgrid(np.arange(ph), np.arange(pw), indexing="ij")
    local = np.concatenate([np.zeros((1, 2), np.int64), np.stack([yy.ravel(), xx.ravel()], 1) + 1], 0)
    glob = np.concatenate([np.zeros((1, 2), np.int64), np.ones((ph * pw, 2), np.int64)], 0)
    return local, glob


def uv_embed(w: int, h: int, aspect: float, ch: int) -> np.ndarray:
    """DualDPT's create_uv_grid + position_grid_to_embed (h, w, ch), times its 0.1."""
    diag = (aspect ** 2 + 1.0) ** 0.5
    sx, sy = aspect / diag, 1.0 / diag
    xs = np.linspace(-sx * (w - 1) / w, sx * (w - 1) / w, w, dtype=np.float32)
    ys = np.linspace(-sy * (h - 1) / h, sy * (h - 1) / h, h, dtype=np.float32)
    uu, vv = np.meshgrid(xs, ys, indexing="xy")
    d = ch // 2

    def sincos(p):
        om = 1.0 / 100.0 ** (np.arange(d // 2, dtype=np.float32) / (d / 2.0))
        o = p.reshape(-1)[:, None] * om[None]
        return np.concatenate([np.sin(o), np.cos(o)], 1)
    return np.concatenate([sincos(uu), sincos(vv)], -1).reshape(h, w, ch) * 0.1


def pose_to_w2c_K(enc: np.ndarray, H: int, W: int):
    """DA3's pose_encoding_to_extri_intri (camera-to-world) then affine_inverse: (w2c S x 3 x 4, K S x 3 x 3)."""
    t, q, fov_h, fov_w = enc[:, :3], enc[:, 3:7], enc[:, 7], enc[:, 8]
    i, j, k, r = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    s = 2.0 / (q * q).sum(-1)
    R = np.stack([1 - s * (j * j + k * k), s * (i * j - k * r), s * (i * k + j * r),
                  s * (i * j + k * r), 1 - s * (i * i + k * k), s * (j * k - i * r),
                  s * (i * k - j * r), s * (j * k + i * r), 1 - s * (i * i + j * j)], -1).reshape(-1, 3, 3)
    Rt = np.transpose(R, (0, 2, 1))
    w2c = np.concatenate([Rt, -Rt @ t[:, :, None]], axis=-1)
    fy = (H / 2.0) / np.maximum(np.tan(fov_h / 2.0), 1e-6)
    fx = (W / 2.0) / np.maximum(np.tan(fov_w / 2.0), 1e-6)
    K = np.zeros((len(enc), 3, 3))
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = fx, fy, W / 2, H / 2, 1.0
    return w2c, K


def ray_pose(ray: np.ndarray, conf: np.ndarray, H: int, W: int, seed: int = 0):
    """Cameras from DA3's ray maps, by DA3's own solver (RANSAC rotation + intrinsics, confidence-weighted origin):
    ray (S, h, w, 6), conf (S, h, w) -> (w2c S x 3 x 4, K S x 3 x 3). Runs PyTorch on the CPU; seeded, since the
    solver samples."""
    import torch
    from depth_anything_3.utils.geometry import affine_inverse
    from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray
    g = torch.random.get_rng_state()
    try:
        for attempt in range(4):
            # a RANSAC sample can be degenerate (DA3's SVD then fails to converge; seen with fp16 rays from BASE):
            # draw the next seed's samples, as rerunning the solver would
            torch.manual_seed(seed + attempt)
            try:
                ext, focal, pp = get_extrinsic_from_camray(torch.from_numpy(ray[None]).float(),
                                                           torch.from_numpy(conf[None]).float(),
                                                           ray.shape[1], ray.shape[2])
                break
            except torch.linalg.LinAlgError:
                if attempt == 3:
                    raise
    finally:
        torch.random.set_rng_state(g)
    # the solver gives camera-to-world ([R | ray origin]); DA3 reports its inverse as the extrinsics
    w2c = affine_inverse(ext)[0, :, :3].numpy().astype(np.float64)
    K = np.tile(np.eye(3), (ray.shape[0], 1, 1))
    f, p = focal[0].numpy(), pp[0].numpy()
    K[:, 0, 0], K[:, 1, 1] = f[:, 0] / 2 * W, f[:, 1] / 2 * H
    K[:, 0, 2], K[:, 1, 2] = p[:, 0] * W * 0.5, p[:, 1] * H * 0.5
    return w2c, K


def _cubic(t, A=-0.75):
    """PyTorch's bicubic convolution weights for offsets t in [0, 1): taps at -1, 0, 1, 2."""
    def near(x):
        return ((A + 2) * x - (A + 3)) * x * x + 1

    def far(x):
        return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A
    return np.stack([far(t + 1), near(t), near(1 - t), far(2 - t)], -1)


def _resample_axis(x, axis, out, scale):
    """torch bicubic along one axis, align_corners=False, output size `out`, coordinates from `scale` (DA3 passes a
    scale factor, so PyTorch maps coordinates by it rather than by the size ratio)."""
    n = x.shape[axis]
    src = (np.arange(out) + 0.5) / scale - 0.5
    i0 = np.floor(src).astype(np.int64)
    w = _cubic((src - i0).astype(np.float64))
    idx = np.clip(i0[:, None] + np.arange(-1, 3)[None], 0, n - 1)
    g = np.take(x, idx.ravel(), axis=axis)
    shp = list(x.shape)
    shp[axis:axis + 1] = [out, 4]
    g = g.reshape(shp)
    wshape = [1] * len(shp)
    wshape[axis], wshape[axis + 1] = out, 4
    return (g * w.reshape(wshape)).sum(axis=axis + 1)


def pos_table(pos_embed: np.ndarray, ph: int, pw: int, offset: float = 0.1) -> np.ndarray:
    """DinoV2's interpolate_pos_encoding (bicubic, DA3's 0.1 offset) for a ph x pw patch grid: (1, 1+ph*pw, C)."""
    cls, grid = pos_embed[:, :1], pos_embed[0, 1:].astype(np.float64)
    M = int(round(np.sqrt(grid.shape[0])))
    g = grid.reshape(M, M, -1)
    g = _resample_axis(g, 0, ph, (ph + offset) / M)
    g = _resample_axis(g, 1, pw, (pw + offset) / M)
    return np.concatenate([cls, g.reshape(1, ph * pw, -1).astype(np.float32)], 1)
