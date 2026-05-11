from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="forestkernel",
    version="0.1.2",
    description="Forest kernel package",
    author="Jake Rhodes, Adrien Aumon",
    author_email="jakerhodes8@gmail.com, adrien.aumon@umontreal.ca",
    url="https://github.com/jakerhodes/RF-GAP-Python",
    packages=find_packages(include=["forestkernel", "forestkernel.*"]),
    install_requires=['scikit-learn', 'numpy', 'pandas', 'scipy', 'packaging', 'lightgbm', 'xgboost', 'pytest', 'pytest-cov'
                      'pyarrow', 'umap-learn', 'matplotlib', 'seaborn', 'joblib'  # for experiments only
                      ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)