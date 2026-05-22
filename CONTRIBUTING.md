# Contributing

Thank you for contributing to `forestgeom`.

## Clone the repository

```bash
git clone https://github.com/jakerhodes/RF-GAP-Python.git
cd RF-GAP-Python
```

## Create a virtual environment

Create and activate a local virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Upgrade packaging tools:

```bash
pip install -U pip setuptools wheel
```

## Install the project

Install the project in editable mode so local source changes are immediately importable:

```bash
pip install -e .
```

Optional extras can be installed as needed:

```bash
# boosted tree support
pip install -e '.[boosted]'

# visualization dependencies
pip install -e '.[viz]'

# testing dependencies
pip install -e '.[test]'
```

## Run tests

```bash
pytest
```

## Notes

- Use a virtual environment for development work.
- Editable installs (`-e`) are recommended for contributors.
- Some optional dependencies (e.g. `lightgbm` or `xgboost`) may require
  additional system-level installation steps. Refer to the corresponding
  project documentation if needed.