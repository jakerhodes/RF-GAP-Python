import numpy as np
import pytest

from forestgeom.encoder import LeafEncoder


class FakeEstimator:
    """A minimal unsupported estimator used to trigger make_adapter TypeError."""
    pass


def test_fit_forest_raises_typeerror_for_unsupported_estimator():
    X = np.zeros((10, 2), dtype=np.float32)
    y = np.arange(10)

    enc = LeafEncoder(forest=FakeEstimator())

    with pytest.raises(TypeError) as exc:
        enc._fit_forest(X, y)

    assert "Unsupported forest estimator" in str(exc.value)
