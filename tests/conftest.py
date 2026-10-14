import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {"da3-small": "depth-anything/DA3-SMALL", "da3-base": "depth-anything/DA3-BASE",
          "da3-large": "depth-anything/DA3-LARGE-1.1"}
CASES = [n for n in MODELS if os.path.exists(f"{ROOT}/ref/{n}.npz")]
BACKENDS = ["mlx", "torch"]


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


_engines = {}


def engine(name, backend, half):
    """One engine per (model, backend, precision) for the whole session: loading LARGE twice would swap."""
    from da3_engine import Engine
    key = (name, backend, half)
    if key not in _engines:
        _engines.clear()
        _engines[key] = Engine(MODELS[name], backend=backend, half=half)
    return _engines[key]


@pytest.fixture(scope="session")
def refs():
    return {n: dict(np.load(f"{ROOT}/ref/{n}.npz")) for n in CASES}
