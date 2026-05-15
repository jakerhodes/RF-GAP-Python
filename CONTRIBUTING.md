# Contributing

Developer setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
```

2. Install the project in editable mode so local edits are importable:

```bash
pip install -e .           # development install
# or install with extras (example)
pip install -e '.[boosted]'
```

3. Run tests:

```bash
pip install -e '.[test]'
pytest
```

Quick GitHub install (for users)

```bash
pip install git+https://github.com/jakerhodes/RF-GAP-Python.git
```

Notes

- Use a virtualenv for project work to avoid polluting system packages.
- If native dependencies are required for extras (e.g., `lightgbm` or `xgboost`), follow those projects' install instructions.
