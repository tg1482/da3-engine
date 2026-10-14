"""The K/V-cached step (context + extend): same on both backends, and how far it is from exact DA3."""
import numpy as np
import pytest

from conftest import CASES, engine, rel


def kv_and_exact(eng, rgb):
    views = [eng.view(im) for im in rgb]
    ctx, _ = eng.context(views[:-1])
    kv = eng.extend(ctx, views[-1], depth=True)
    exact = eng.infer(views, depth=[len(views) - 1])
    return kv, exact


@pytest.mark.parametrize("name", CASES)
def test_extend_same_on_both_backends(name, refs):
    """Two independent implementations of one definition agree (float32)."""
    rgb = refs[name]["rgb"]
    a, _ = kv_and_exact(engine(name, "mlx", False), rgb)
    b, _ = kv_and_exact(engine(name, "torch", False), rgb)
    n = len(rgb) - 1
    assert rel(a.depth[n], b.depth[n]) < 1e-3
    assert np.abs(a.w2c - b.w2c).max() < 1e-3


@pytest.mark.parametrize("name", CASES)
def test_extend_context_cameras_are_the_contexts(name, refs):
    """The context's cameras in an extend step are exactly those of the context run (kept, not recomputed)."""
    eng = engine(name, "mlx", False)
    views = [eng.view(im) for im in refs[name]["rgb"]]
    ctx, r0 = eng.context(views[:-1])
    r1 = eng.extend(ctx, views[-1])
    assert np.array_equal(r1.w2c[:-1], r0.w2c)


def rel_t(w2c, a, b):
    """Translation of camera b relative to camera a."""
    A, B = np.vstack([w2c[a], [0, 0, 0, 1]]), np.vstack([w2c[b], [0, 0, 0, 1]])
    return (B @ np.linalg.inv(A))[:3, 3]


@pytest.mark.parametrize("name", CASES)
def test_extend_vs_exact_report(name, refs):
    """How far the K/V step is from exact, next to how far exact DA3 itself moves when one view joins its set
    (measured 10-14: kv 0.08-0.11 translation / 3-7% depth; one view more 0.09-0.27 / 8-15%, units of DA3's
    scene). Not a threshold: whether it is good enough is the SLAM bench's call. Asserted: it is sane (the error
    is well below the motion itself)."""
    eng = engine(name, "mlx", False)
    v = [eng.view(im) for im in refs[name]["rgb"]]
    n = len(v) - 1
    e6, e5 = eng.infer(v, depth=[n - 1, n]), eng.infer(v[:n], depth=[n - 1])
    ctx, _ = eng.context(v[:n])
    kv = eng.extend(ctx, v[n])
    kv_t = np.linalg.norm(rel_t(kv.w2c, 0, n) - rel_t(e6.w2c, 0, n))
    kv_d = rel(kv.depth[n], e6.depth[n])
    own_t = np.linalg.norm(rel_t(e6.w2c, 0, n - 1) - rel_t(e5.w2c, 0, n - 1))
    own_d = rel(e5.depth[n - 1], e6.depth[n - 1])
    print(f"{name}: kv vs exact t {kv_t:.3f} depth {kv_d:.3f} | exact, one view more: t {own_t:.3f} depth {own_d:.3f}")
    assert kv_t < 0.5 * np.linalg.norm(rel_t(e6.w2c, 0, n)) and kv_d < 0.5
