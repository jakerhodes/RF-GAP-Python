# forestgeom

`forestgeom` builds sparse leaf-incidence representations for tree ensembles and
uses them to compute forest kernels/proximities. The core API is
`forestgeom.LeafEncoder`, which fits a supported ensemble, encodes samples by
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

# Installation (recommended)
The recommended way for most users to install is directly from the GitHub repository into a virtual environment. This lets users install the package into their own venv without waiting for a PyPI release.

```bash
# install the latest main branch into the active venv
pip install git+https://github.com/jakerhodes/RF-GAP-Python.git

# or install a specific branch or tag
pip install git+https://github.com/jakerhodes/RF-GAP-Python.git@main
```

GitHub install examples

```bash
# install with an extras group (boosted)
pip install 'git+https://github.com/jakerhodes/RF-GAP-Python.git@main#egg=forestgeom[boosted]'

# editable VCS install (development) with extras
pip install -e 'git+https://github.com/jakerhodes/RF-GAP-Python.git@main#egg=forestgeom[boosted]'

# install from GitHub but skip automatic dependency installation
pip install --no-deps 'git+https://github.com/jakerhodes/RF-GAP-Python.git'

# use a constraints file to control versions when installing from GitHub
pip install 'git+https://github.com/jakerhodes/RF-GAP-Python.git' -c constraints.txt
```

Local / development install

If you're developing on the project or want an editable install (changes in the checkout are immediately importable), use a virtualenv and one of the following:

```bash
# normal local install
pip install .

# editable / development install (recommended for contributors)
pip install -e .
```

# Extras (optional dependencies)

Install optional feature groups only when needed:

```bash
# install boosted extras
pip install -e '.[boosted]'

# install viz extras
pip install -e '.[viz]'
```

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
from forestgeom import LeafEncoder

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

For proximity-weighted prediction, `LeafEncoder.predict(X)` and
`LeafEncoder.predict_proba(X)` provide matrix-free convenience wrappers that use
the fitted base forest task type: regression forests return weighted responses,
while classification forests return weighted class predictions/probabilities.
They avoid materializing the full proximity matrix `P` by multiplying the sparse
leaf factors against the training targets or class indicators directly.

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
