import pandas as pd
import pytest

from denseclus.utils import extract_categorical, extract_numerical, transform_numerics

# TO DO: Parameterize to conftest
cat_df = pd.DataFrame({"col1": ["A", "B", "A", "B"], "col2": ["C", "D", "E", "C"]})
num_df = pd.DataFrame({"col1": [23.0, 43.0, 50.0], "col2": [33.0, 34.0, 55.0]})


def test_extract_categorical():
    categories = extract_categorical(cat_df)
    assert categories.shape == (4, 5)
    assert categories.min().min() == 0
    assert categories.max().min() == 1


def test_extract_numerical():
    numerics = extract_numerical(num_df)
    assert str(numerics["col1"].dtypes) == "float64"
    assert str(numerics["col2"].dtypes) == "float64"


def test_extract_categorical_is_df():
    with pytest.raises(TypeError):
        extract_categorical(["A", "B", "C"])


def test_extract_categorical_is_object():
    with pytest.raises(ValueError):
        extract_categorical(num_df)


def test_extract_numerical_is_df():
    with pytest.raises(TypeError):
        extract_numerical([1, 2, 3])


def test_extract_numerical_is_numeric():
    with pytest.raises(ValueError):
        extract_numerical(cat_df)


def test_transform_numerics():
    numerics = transform_numerics(num_df)
    assert len(numerics) == 3


def test_transform_numerics_is_df():
    with pytest.raises(TypeError):
        transform_numerics([1, 2, 3])
