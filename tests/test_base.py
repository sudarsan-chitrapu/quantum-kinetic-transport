import numpy as np
import pytest

import models.base as base


def test_base_is_abstract():
    assert issubclass(base.BlochModel, base.ABC)

    with pytest.raises(TypeError):
        base.BlochModel(dimension=2, n_bands=3)


def test_concrete_model_implements_required_methods():
    class ToyModel(base.BlochModel):
        def H(self, k: np.ndarray) -> np.ndarray:
            return np.zeros((3, 3), dtype=complex)

        def dH_dk(self, k: np.ndarray) -> np.ndarray:
            return np.zeros((2, 3, 3), dtype=complex)

    model = ToyModel(dimension=2, n_bands=3)

    assert model.dimension == 2
    assert model.n_bands == 3
    assert model.H(np.zeros(2)).shape == (3, 3)
    assert model.dH_dk(np.zeros(2)).shape == (2, 3, 3)