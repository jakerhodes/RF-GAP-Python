from setuptools import setup, find_packages

with open("README.md", "r", encoding = "utf-8") as f:
    long_description = f.read()

install_requires = ['scikit-learn', 'numpy', 'pandas', 'scipy', 'packaging']

setup(
    name = 'rfgap',
    version = '0.0.1',
    author = 'Jake Rhodes',
    author_email = 'jakerhodes8@gmail.com',
    description = 'RF-GAP Proximities - Python',
    long_description = long_description,
    url = 'https://github.com/jakerhodes/RF-GAP-Python',
    license = 'GNU-V3',
    packages=find_packages(include=["rfgap", "rfgap.*"]),
    install_requires = install_requires
)