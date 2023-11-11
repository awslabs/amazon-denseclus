"""
    Fixture configs for tests
"""

import pandas as pd
import pytest

from denseclus.DenseClus import DenseClus
from denseclus.utils import make_dataframe


@pytest.fixture
def n_components():
    return 3


@pytest.fixture
def df():
    df = make_dataframe()
    return df


@pytest.fixture
def clf(df, n_components):
    umap_params = {
        "categorical": {"n_components": n_components},
        "numerical": {"n_components": n_components},
        "combined": {"n_components": n_components},
    }
    clf = DenseClus(umap_combine_method="intersection_union_mapper", umap_params=umap_params)
    clf.fit(df)
    return clf


@pytest.fixture
def categorical_df():
    return pd.DataFrame({"col1": ["A", "B", "A", "B"], "col2": ["C", "D", "E", "C"]})


@pytest.fixture
def numerical_df():
    return pd.DataFrame({"col3": [23.0, 43.0, 50.0], "col4": [33.0, 34.0, 55.0]})
