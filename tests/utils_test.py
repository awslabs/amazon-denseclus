import pandas as pd
import pytest
from denseclus.utils import extract_categorical, extract_numerical, transform_numerics


def test_extract_categorical(categorical_df):
    categories = extract_categorical(categorical_df)
    assert categories.shape == (4, 5)
    assert categories.min().min() == 0
    assert categories.max().min() == 1


def test_extract_numerical(numerical_df):
    numerics = extract_numerical(numerical_df)
    assert str(numerics["col3"].dtypes) == "float64"
    assert str(numerics["col4"].dtypes) == "float64"


def test_extract_categorical_is_df():
    with pytest.raises(AttributeError):
        extract_categorical(["A", "B", "C"])


def test_extract_categorical_is_object(numerical_df):
    with pytest.raises(ValueError):
        extract_categorical(numerical_df)


def test_extract_numerical_is_df():
    with pytest.raises(AttributeError):
        extract_numerical([1, 2, 3])


def test_extract_numerical_is_numeric(categorical_df):
    with pytest.raises(ValueError):
        extract_numerical(categorical_df)


def test_transform_numerics(numerical_df):
    numerics = transform_numerics(numerical_df)
    assert len(numerics) == 3


def test_transform_numerics_is_df():
    with pytest.raises(AttributeError):
        transform_numerics([1, 2, 3])


def test_extract_categorical_empty_df():
    with pytest.raises(ValueError):
        extract_categorical(pd.DataFrame())


def test_extract_numerical_empty_df():
    with pytest.raises(ValueError):
        extract_numerical(pd.DataFrame())
