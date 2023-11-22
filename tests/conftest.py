#!/usr/bin/env python3
"""
    Fixture configs for tests
"""

import pandas as pd
import numpy as np
import pytest
from denseclus.DenseClus import DenseClus
from denseclus.utils import make_dataframe


N_COMPONENTS = 3
DF_LEN = 5


@pytest.fixture(scope="module")
def df_len():
    """Length of new df for fit predict"""
    return DF_LEN


@pytest.fixture(scope="module")
def df():
    df = make_dataframe()
    return df


@pytest.fixture(scope="module")
def fitted_clf():
    clf = DenseClus()
    df_small = make_dataframe(n_samples=100)
    clf.fit(df_small)
    return clf


@pytest.fixture(scope="module")
def fitted_predictions(default_clf, df, df_len):
    preds = default_clf.fit_predict(df, df.tail(df_len))
    return preds


@pytest.fixture(scope="module")
def union_mapper_clf():
    df = make_dataframe()
    umap_params = {
        "categorical": {"n_components": N_COMPONENTS},
        "numerical": {"n_components": N_COMPONENTS},
        "combined": {"n_components": N_COMPONENTS},
    }
    clf = DenseClus(umap_combine_method="intersection_union_mapper", umap_params=umap_params)
    clf.fit(df)
    return clf


@pytest.fixture(scope="module")
def default_clf():
    df = make_dataframe()
    clf = DenseClus()
    clf.fit(df)
    return clf


@pytest.fixture(scope="module")
def categorical_df():
    return pd.DataFrame({"col1": ["A", "B", "A", "B"], "col2": ["C", "D", "E", "C"]})


@pytest.fixture(scope="module")
def numerical_df():
    return pd.DataFrame({"col3": [23.0, 43.0, 50.0], "col4": [33.0, 34.0, 55.0]})


@pytest.fixture(scope="module")
def missing_numerical_df():
    df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, 5, np.nan]})
    return df


@pytest.fixture(scope="module")
def missing_categorical_df():
    df = pd.DataFrame({"A": ["alpha", np.nan, "beta"], "B": ["gamma", "delta", np.nan]})
    return df


@pytest.fixture(scope="module")
def df_new():
    return make_dataframe()
