#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pytest
import warnings
from denseclus.DenseClus import DenseClus


def check_embedding_shape(embedding, expected_shape):
    assert embedding.shape == expected_shape


@pytest.mark.slow
def test_fit_categorical(union_mapper_clf, df):
    check_embedding_shape(
        union_mapper_clf.categorical_umap_.embedding_,
        (len(df), union_mapper_clf.categorical_umap_.n_components),
    )


@pytest.mark.slow
def test_fit_numerical(union_mapper_clf, df):
    check_embedding_shape(
        union_mapper_clf.numerical_umap_.embedding_,
        (len(df), union_mapper_clf.numerical_umap_.n_components),
    )


@pytest.mark.slow
def test_umap_embeddings(union_mapper_clf, df):
    assert union_mapper_clf.mapper_.embedding_.shape == (
        len(df),
        union_mapper_clf.mapper_.n_components[-1],
    )


@pytest.mark.slow
def test_hdbscan_labels(union_mapper_clf, df):
    assert union_mapper_clf.hdbscan_.labels_.shape[0] == df.shape[0]


@pytest.mark.slow
def test_denseclus_fit_is_df(union_mapper_clf):
    with pytest.raises(TypeError):
        union_mapper_clf.fit([1, 2, 3])


@pytest.mark.slow
def test_denseclus_score(union_mapper_clf, df):
    assert len(union_mapper_clf.score()) == len(df)


@pytest.mark.fast
def test_denseclus_method(df):
    with pytest.raises(ValueError):
        _ = DenseClus(umap_combine_method="notamethod").fit(df)


@pytest.mark.slow
def test_repr(union_mapper_clf):
    warnings.filterwarnings("ignore", category=UserWarning, module="umap.umap_")
    assert str(type(union_mapper_clf.__repr__)) == "<class 'method'>"


@pytest.mark.fast
def test_denseclus_score_length(fitted_clf):
    scores = fitted_clf.score()
    assert len(scores) == 100


@pytest.mark.fast
def test_denseclus_score_output(fitted_clf):
    scores = fitted_clf.score()
    expected_output = np.array([-1] * 100)
    assert np.all(expected_output == scores)


@pytest.mark.fast
def test_fit_empty_df():
    with pytest.raises(ValueError):
        DenseClus().fit(pd.DataFrame())


@pytest.mark.slow
def test_predict(default_clf, df_new):
    labels = default_clf.predict(df_new)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(df_new)


@pytest.mark.fast
def test_predict_output_shape(default_clf, df_new):
    labels = default_clf.predict(df_new)
    assert labels.shape == (
        len(df_new),
    ), "Predicted labels should have the same number of rows as input data"


@pytest.mark.fast
def test_predict_output_type(default_clf, df_new):
    labels = default_clf.predict(df_new)
    assert issubclass(labels.dtype.type, np.integer), "Predicted labels should be integers"


@pytest.mark.fast
def test_predict_proba_output_range(default_clf, df_new):
    probabilities = default_clf.predict_proba(df_new)
    assert np.all(
        (probabilities >= 0) & (probabilities <= 1),
    ), "Probabilities should be between 0 and 1"


@pytest.mark.fast
def test_predict_proba_output_shape(default_clf, df_new):
    probabilities = default_clf.predict_proba(df_new)
    assert probabilities.shape == (
        len(df_new),
    ), "Probabilities should have the same number of rows as input data"


@pytest.mark.fast
def test_predict_input_type(union_mapper_clf):
    with pytest.raises(TypeError):
        union_mapper_clf.predict("not a dataframe")


@pytest.mark.fast
def test_predict_proba_input_type(union_mapper_clf):
    with pytest.raises(TypeError):
        union_mapper_clf.predict_proba("not a dataframe")
