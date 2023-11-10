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


def test_fit_known_output(df):
    pass
    df_small = df.head(100)
    clf = DenseClus()
    clf.fit(df_small)
    scores = clf.score()
    expected_output = np.array([-1] * 100)
    assert len(clf.numerical_umap_.embedding_) == len(expected_output)
    assert np.all(expected_output == scores)


def test_fit_empty_df():
    with pytest.raises(ValueError):
        DenseClus().fit(pd.DataFrame())
