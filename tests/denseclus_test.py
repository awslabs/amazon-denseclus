#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
import warnings
from denseclus.DenseClus import DenseClus


def test_fit_categorical(n_components, df):
    clf = DenseClus(n_components=n_components)
    clf.fit(df)
    assert clf.categorical_umap_.embedding_.shape == (len(df), n_components)


def test_fit_numerical(clf, df):
    assert clf.numerical_umap_.embedding_.shape == (len(df), clf.n_components)


def test_umap_embeddings(clf, df):
    assert clf.mapper_.embedding_.shape == (len(df), clf.n_components)


def test_hdbscan_labels(clf, df):
    assert clf.hdbscan_.labels_.shape[0] == df.shape[0]


def test_denseclus_fit_is_df(clf):
    with pytest.raises(TypeError):
        clf.fit([1, 2, 3])


def test_denseclus_score(clf, df):
    assert len(clf.score()) == len(df)


def test_denseclus_method(df):
    with pytest.raises(ValueError):
        _ = DenseClus(umap_combine_method="notamethod").fit(df)


def test_repr(clf):
    assert str(type(clf.__repr__)) == "<class 'method'>"


def test_fit_known_output(categorical_df, numerical_df):
    pass
    # df_small = pd.concat([categorical_df, numerical_df])
    # clf.fit(df_small)
    # expected_output = ""
    # assert np.allclose(clf.numerical_umap_.embedding_, expected_output)


def test_fit_empty_df():
    with pytest.raises(OverflowError):
        DenseClus().fit(pd.DataFrame())
