import numpy as np
import pytest

from forestgeom import LeafEncoder

from tests.fixtures.constants import BOOSTED_FORESTS_AND_DATA


if BOOSTED_FORESTS_AND_DATA:

    @pytest.mark.integration
    @pytest.mark.parametrize("forest_fixture,data_fixture", BOOSTED_FORESTS_AND_DATA)
    def test_boosted_kernel_matches_explicit_weighted_leaf_collisions(
        request,
        forest_fixture,
        data_fixture,
    ):
        X_train, _, y_train, _ = request.getfixturevalue(data_fixture)
        forest = request.getfixturevalue(forest_fixture)

        enc = LeafEncoder(forest=forest, weight_scheme="boosted").fit(X_train, y_train)

        K_fast = enc.proximity(return_dense=True)
        leaves = enc.forest_.get_leaf_matrix(X_train)
        weights = enc.forest_.get_tree_weights(X_train)

        n_samples, n_trees = leaves.shape
        K_slow = np.zeros((n_samples, n_samples), dtype=np.float32)

        for t in range(n_trees):
            same_leaf = leaves[:, [t]] == leaves[:, [t]].T
            K_slow += weights[t] * same_leaf.astype(np.float32)

        assert np.allclose(K_fast, K_slow, rtol=1e-5, atol=1e-6)
