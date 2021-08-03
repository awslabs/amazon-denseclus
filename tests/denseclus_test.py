#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from denseclus.DenseClus import DenseClus

# TO DO: Parameterize in conftest
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

clf = DenseClus(
    n_components=3,
    random_state=42,
    n_neighbors=10,
    umap_combine_method="intersection_union_mapper",
)
clf.fit(df)


def test_fit_categorical():
    assert clf.categorical_umap_.embedding_.shape == (len(df), clf.n_components)


def test_fit_numerical():
    assert clf.numerical_umap_.embedding_.shape == (len(df), clf.n_components)


def test_umap_embeddings():
    assert clf.mapper_.embedding_.shape == (len(df), clf.n_components)


def test_hdbscan_labels():
    assert clf.hdbscan_.labels_.shape[0] == df.shape[0]


def test_denseclus_fit_is_df():
    with pytest.raises(TypeError):
        clf.fit([1, 2, 3])


def test_denseclus_score():
    assert len(clf.score()) == len(df)


def test_denseclus_method():
    with pytest.raises(KeyError):
        _ = DenseClus(umap_combine_method="notamethod").fit(df)


def test_repr():
    assert str(type(clf.__repr__)) == "<class 'method'>"
