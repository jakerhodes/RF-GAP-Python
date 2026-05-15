# forestkernel

`forestkernel` builds sparse leaf-incidence representations for tree ensembles and
uses them to compute forest kernels/proximities. The core API is
`forestkernel.LeafEncoder`, which fits a supported ensemble, encodes samples by
the leaves they reach, and represents the train-train kernel in factored form:

```text
P = Q W^T
```

Here, `Q` is the query-side leaf map and `W` is the reference-side leaf map. Both
maps are sparse, with at most one nonzero per tree per sample, so downstream code
can work directly with the sparse factors instead of materializing a dense
pairwise proximity matrix.

The package implements forest proximity constructions described in
“Random Forest- Geometry- and Accuracy-Preserving Proximities”
(https://ieeexplore.ieee.org/document/10089875) and
“Revisiting Forest Proximities via Sparse Leaf-Incidence Kernels”
(https://arxiv.org/abs/2601.02735).

# Installation
To install, please use ```pip install git+https://github.com/jakerhodes/RF-GAP-Python```

# Usage

`LeafEncoder` wraps a tree ensemble estimator and clones/fits it during
`fit(...)`. It supports scikit-learn Random Forests, ExtraTrees, and Gradient
Boosting estimators, with optional adapters for LightGBM and XGBoost when those
packages are installed.

Supported leaf-weighting schemes include:

- `uniform`: symmetric leaf co-occurrence factorization of the standard forest
  kernel.
- `kerf`: symmetric leaf-size-normalized factorization of the KeRF kernel.
- `oob`: separable OOB leaf-incidence factorization that approximates the
  off-diagonal Breiman OOB affinities.
- `gap`: asymmetric query/reference factorization that combines OOB-side query
  weights with in-bag reference weights to recover the GAP proximity definition.
- `boosted`: symmetric tree-weighted leaf kernel for supported boosted
  ensembles.

Not every estimator supports every weighting scheme. Random Forests and
ExtraTrees estimators support `uniform` and `kerf`; they support `oob` and
`gap` only when fitted with `bootstrap=True`. Boosted estimators support
`uniform`, `kerf`, and `boosted`.

```python
from sklearn.ensemble import RandomForestClassifier
from forestkernel import LeafEncoder

forest = RandomForestClassifier(
    n_estimators=500,
    bootstrap=True,
    random_state=0,
    n_jobs=-1,
)

encoder = LeafEncoder(forest=forest, weight_scheme="gap").fit(X_train, y_train)

# Sparse leaf maps for custom downstream work.
Q_train = encoder.training_query_map()
W_train = encoder.reference_map()
Q_test = encoder.transform(X_test)

# Explicit kernel/proximity matrices. These are sparse by default.
K_train = encoder.kernel()
K_test_train = encoder.kernel_extend(X_test)

# Dense output is available when needed.
K_train_dense = encoder.kernel(return_dense=True)
```

For asymmetric weighting schemes such as `gap`, the fitted training query map
`Q_train` and reference map `W_train` differ. The train-train kernel is still
computed as `Q_train @ W_train.T`, and `kernel_extend(X)` returns
`Q(X) @ W_train.T` for out-of-sample data.

The sparse factors can be used directly in kernel methods, manifold learning,
dimensionality reduction, visualization, imputation, and other proximity-based
workflows.

# Citation

If you use this software in your research or experiments, please cite the following works:

```bibtex
@ARTICLE{10089875,
  author={Rhodes, Jake S. and Cutler, Adele and Moon, Kevin R.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Geometry- and Accuracy-Preserving Random Forest Proximities}, 
  year={2023},
  volume={45},
  number={9},
  pages={10947-10959},
  keywords={Random forests;Forestry;Geometry;Data visualization;Decision trees;Task analysis;Anomaly detection;Proximities;random forests;supervised learning},
  doi={10.1109/TPAMI.2023.3263774}}
```

```bibtex
@misc{aumon2026revisitingforestproximitiessparse,
      title={Revisiting Forest Proximities via Sparse Leaf-Incidence Kernels}, 
      author={Adrien Aumon and Guy Wolf and Kevin R. Moon and Jake S. Rhodes},
      year={2026},
      eprint={2601.02735},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.02735}, 
}
```
