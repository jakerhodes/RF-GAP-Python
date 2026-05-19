import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=150,
        n_classes=3,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=0,
    )
    return train_test_split(X, y, test_size=0.25, random_state=0)


@pytest.fixture
def regression_data():
    X, y = make_regression(
        n_samples=150,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=0,
    )
    return train_test_split(X, y, test_size=0.25, random_state=0)


@pytest.fixture
def classification_data_10k():
    X, y = make_classification(
        n_samples=10_000,
        n_classes=3,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=0,
    )
    return train_test_split(X, y, test_size=0.25, random_state=0)


@pytest.fixture
def regression_data_10k():
    X, y = make_regression(
        n_samples=10_000,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=0,
    )
    return train_test_split(X, y, test_size=0.25, random_state=0)
