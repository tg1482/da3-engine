"""The engine against DA3 itself (tools/export_reference.py: PyTorch DepthAnything3Net, float32), per backend."""
import numpy as np
import pytest

from conftest import BACKENDS, CASES, engine, rel
from da3_engine.common import pos_table, prep


def test_prep_matches_da3_input_processor():
    torch = pytest.importorskip("torch")
    from depth_anything_3.utils.io.input_processor import InputProcessor
    rng = np.random.default_rng(0)
    for h, w in ((480, 640), (200, 300), (378, 504), (720, 1280)):        # down, up, as-is, 16:9
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        ref, _, _ = InputProcessor()([img], None, None, 504, "upper_bound_resize")
        assert np.abs(prep(img, 504) - ref[0].permute(1, 2, 0).numpy()).max() < 1e-5


@pytest.mark.parametrize("name", CASES)
def test_pos_table_matches_torch_bicubic(name, refs):
    import safetensors.numpy
    from da3_engine.engine import default_weights_dir
    from conftest import MODELS
    engine(name, "mlx", False)                                           # converts the weights if needed
    pe = safetensors.numpy.load_file(f"{default_weights_dir(MODELS[name])}/weights.safetensors")["pos_embed"]
    H, W = refs[name]["images"].shape[1:3]
    assert np.abs(pos_table(pe, H // 14, W // 14) - refs[name]["pos_table"]).max() < 1e-5


def to_np(x):
    if hasattr(x, "cpu"):
        return x.float().cpu().numpy()
    import mlx.core as mx
    return np.array(x.astype(mx.float32))


def run_exact(eng, ref):
    views = [eng.view(im) for im in ref["rgb"]]
    S = len(views)
    return views, eng.infer(views, depth=range(S), rays=range(S), ray_pose=True)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", CASES)
def test_float32_matches_da3(name, backend, refs):
    ref = refs[name]
    eng = engine(name, backend, False)
    views, r = run_exact(eng, ref)
    S = len(views)
    assert rel(to_np(views[0].tokens["x"][0]), ref["early_x"][0]) < 1e-4
    assert rel(np.stack([r.depth[i] for i in range(S)]), ref["depth"]) < 1e-3
    assert rel(np.stack([r.depth_conf[i] for i in range(S)]), ref["conf"]) < 1e-3
    assert rel(np.stack([r.ray[i] for i in range(S)]), ref["ray"]) < 1e-3
    assert rel(np.stack([r.ray_conf[i] for i in range(S)]), ref["ray_conf"]) < 1e-3
    assert np.abs(r.w2c - ref["w2c"]).max() < 1e-3
    assert np.abs(r.K - ref["K"]).max() / ref["K"].max() < 1e-3
    # the ray solver samples (seeded alike); its inputs agree to ~1e-5, so its poses to well under a millimetre
    assert np.abs(np.stack([r.ray_w2c[i] for i in range(S)]) - ref["ray_w2c"]).max() < 1e-2


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", CASES)
def test_float16_close_to_da3(name, backend, refs):
    """Half precision (as the SLAM server runs): depth and rays within 1%, camera-head poses within 1 cm."""
    ref = refs[name]
    eng = engine(name, backend, True)
    views = [eng.view(im) for im in ref["rgb"]]
    S = len(views)
    r = eng.infer(views, depth=range(S), rays=range(S), ray_pose=True)
    assert rel(np.stack([r.depth[i] for i in range(S)]), ref["depth"]) < 1e-2
    assert rel(np.stack([r.ray[i] for i in range(S)]), ref["ray"]) < 1e-2
    assert np.abs(r.w2c - ref["w2c"]).max() < 1e-2


@pytest.mark.parametrize("backend", BACKENDS)
def test_heads_for_some_views_equal_all(backend, refs):
    """Depth for one view (what a SLAM step asks for) equals that view from a run asking for all."""
    name = CASES[0]
    eng = engine(name, backend, False)
    views = [eng.view(im) for im in refs[name]["rgb"]]
    one = eng.infer(views, depth=[5])
    all_ = eng.infer(views, depth=range(6))
    assert np.allclose(one.depth[5], all_.depth[5], rtol=1e-4, atol=1e-5)
