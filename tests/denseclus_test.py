#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pytest
import warnings
from denseclus.DenseClus import DenseClus

LABELS = 1000  # make_dataframe default of 1000


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


@pytest.mark.fast
def test_denseclus_method(df):
    with pytest.raises(ValueError):
        _ = DenseClus(umap_combine_method="notamethod").fit(df)


@pytest.mark.slow
def test_repr(union_mapper_clf):
    warnings.filterwarnings("ignore", category=UserWarning, module="umap.umap_")
    assert str(type(union_mapper_clf.__repr__)) == "<class 'method'>"


@pytest.mark.fast
def test_denseclus_evalute_length(fitted_clf):
    scores = fitted_clf.evaluate()
    assert len(scores) == LABELS


@pytest.mark.fast
def test_denseclus_evalute_output(fitted_clf):
    scores = fitted_clf.evaluate()
    assert 1 in scores
    assert 0 in scores
    assert isinstance(scores, np.ndarray)


@pytest.mark.fast
def test_fit_empty_df():
    with pytest.raises(ValueError):
        DenseClus().fit(pd.DataFrame())


@pytest.mark.slow
def test_predict(fitted_predictions, df_len):
    labels = fitted_predictions[:, 0]
    assert isinstance(labels, np.ndarray)
    assert len(labels) == df_len


@pytest.mark.fast
def test_predict_output_shape(fitted_predictions, df_len):
    labels = fitted_predictions[:, 0]
    assert labels.shape == (
        df_len,
    ), "Predicted labels should have the same number of rows as input data"


@pytest.mark.fast
def test_predict_output_type(fitted_predictions):
    labels = fitted_predictions[:, 0]
    assert issubclass(labels.dtype.type, np.integer) or issubclass(
        labels.dtype.type,
        np.float64,
    ), "Predicted labels should be integers"


@pytest.mark.fast
def test_predict_proba_output_range(fitted_predictions):
    probabilities = fitted_predictions[:, 1]
    assert np.all(
        (probabilities >= 0) & (probabilities <= 1),
    ), "Probabilities should be between 0 and 1"


@pytest.mark.fast
def test_predict_proba_output_shape(fitted_predictions, df_len):
    probabilities = fitted_predictions[:, 1]
    assert probabilities.shape == (
        df_len,
    ), "Probabilities should have the same number of rows as input data"


@pytest.mark.fast
def test_predict_input_type(fitted_clf):
    with pytest.raises(TypeError):
        fitted_clf.predict("not a dataframe")


@pytest.mark.fast
def test_mapper_property(union_mapper_clf):
    try:
        _ = union_mapper_clf.mapper_
    except AttributeError:
        pytest.fail("dense_clus.mapper_ raised AttributeError unexpectedly!")


@pytest.mark.fast
def test_umap_combine_method_property(fitted_clf):
    try:
        fitted_clf.umap_combine_method = "intersection"
    except ValueError:
        pytest.fail("Setting valid umap_combine_method raised ValueError unexpectedly!")

    with pytest.raises(ValueError):
        fitted_clf.umap_combine_method = "invalid_method"


@pytest.mark.fast
def test_random_state_property(fitted_clf):
    try:
        fitted_clf.random_state = 123
    except ValueError:
        pytest.fail("Setting valid random_state raised ValueError unexpectedly!")

    with pytest.raises(ValueError):
        fitted_clf.random_state = "invalid_state"


def test_evaluate_mapper(union_mapper_clf):
    labels = union_mapper_clf.evaluate()

    assert isinstance(labels, np.ndarray)
    assert len(labels) == LABELS
    assert len(labels) == LABELS
    assert len(labels) == LABELS


def test_evaluate(fitted_clf):
    labels = fitted_clf.evaluate()

    assert isinstance(labels, np.ndarray)
    assert len(labels) == LABELS
    assert len(labels) == LABELS
