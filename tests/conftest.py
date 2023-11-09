"""
    Fixture configs for tests
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from denseclus.DenseClus import DenseClus
import warnings


@pytest.fixture(params=[1, 2, 3, 10])
def n_components(request):
    return request.param


@pytest.fixture
def df():
    n_clusters = 3
    X, y = make_blobs(n_samples=1000, n_features=8, random_state=10)
    numerical = StandardScaler().fit_transform(X[:, :6])
    categorical = KBinsDiscretizer(n_bins=3, encode="ordinal").fit_transform(X[:, 6:])
    categorical = np.where(
        categorical == 1.0,
        "M",
        np.where(categorical == 2.0, "H", "L"),
    ).astype(str)

    numerical_columns = [f"num_{i}" for i in range(numerical.shape[1])]
    df = pd.DataFrame(numerical, columns=numerical_columns)

    categorical_columns = [f"cat_{i}" for i in range(categorical.shape[1])]
    for idx, c in enumerate(categorical_columns):
        df[c] = categorical[:, idx]

    return df


@pytest.fixture
def clf(df):
    clf = DenseClus(
        n_components=3,
        random_state=42,
        n_neighbors=10,
        umap_combine_method="intersection_union_mapper",
    )
    clf.fit(df)
    return clf


@pytest.fixture
def categorical_df():
    return pd.DataFrame({"col1": ["A", "B", "A", "B"], "col2": ["C", "D", "E", "C"]})


@pytest.fixture
def numerical_df():
    return pd.DataFrame({"col3": [23.0, 43.0, 50.0], "col4": [33.0, 34.0, 55.0]})
