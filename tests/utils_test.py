import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from denseclus.utils import (
    extract_categorical,
    extract_numerical,
    transform_numerics,
    impute_categorical,
    impute_numerical,
)


def test_extract_categorical(categorical_df):
    categories = extract_categorical(categorical_df)
    assert categories.shape == (4, 3)
    assert categories.min().min() == 0
    assert categories.max().min() == 1


def test_extract_numerical(numerical_df):
    numerics = extract_numerical(numerical_df)
    assert str(numerics["col3"].dtypes) == "float64"
    assert str(numerics["col4"].dtypes) == "float64"


def test_extract_categorical_is_df():
    with pytest.raises(TypeError):
        extract_categorical(["A", "B", "C"])


def test_extract_categorical_is_object(numerical_df):
    with pytest.raises(ValueError):
        extract_categorical(numerical_df)


def test_extract_numerical_is_df():
    with pytest.raises(TypeError):
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


def test_extract_categorical_no_categorical(categorical_df):
    df = categorical_df.select_dtypes(include=["float", "int"])
    with pytest.raises(ValueError):
        extract_categorical(df)


def test_extract_categorical_with_categorical(categorical_df):
    result = extract_categorical(categorical_df)
    expected = pd.DataFrame(
        {
            "col1_B": [False, True, False, True],
            "col2_D": [False, True, False, False],
            "col2_E": [False, False, True, False],
        },
    )
    assert_frame_equal(result, expected)


def test_extract_categorical_with_high_cardinality(categorical_df):
    df = pd.DataFrame(
        {"col1": ["A", "B", "A", "B"], "col2": ["C", "D", "E", "C"], "col3": ["A", "A", "B", "B"]},
    )
    result = extract_categorical(df, cardinality_threshold=1)
    assert result.sum(axis=0).sum() == 0
    assert result.sum(axis=1).max(axis=0) == 1.0
    assert result.shape == (4, 3)


def test_impute_categorical_no_missing_values(categorical_df):
    result = impute_categorical(categorical_df, strategy="constant", fill_value="Missing")
    assert_frame_equal(result, categorical_df)


def test_impute_categorical_with_missing_values():
    df = pd.DataFrame({"col1": ["A", "B", np.nan, "B"], "col2": ["C", np.nan, "E", "C"]})
    expected = pd.DataFrame(
        {"col1": ["A", "B", "Missing", "B"], "col2": ["C", "Missing", "E", "C"]},
    )
    result = impute_categorical(df, strategy="constant", fill_value="Missing")
    assert_frame_equal(result, expected)


def test_impute_categorical_with_custom_strategy():
    df = pd.DataFrame({"col1": ["A", "B", np.nan, "B"], "col2": ["C", np.nan, "E", "C"]})
    expected = pd.DataFrame({"col1": ["A", "B", "Z", "B"], "col2": ["C", "Z", "E", "C"]})
    result = impute_categorical(
        df,
        strategy="constant",
        fill_value="Missing",
        custom_strategy=lambda s: "Z",
    )
    assert_frame_equal(result, expected)


def test_extract_categorical_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame should not be empty."):
        extract_categorical(df)


def test_extract_categorical_no_columns():
    df = pd.DataFrame(index=[0, 1, 2])
    with pytest.raises(ValueError, match="Input DataFrame should not be empty."):
        extract_categorical(df)


def test_extract_categorical_no_categorical_data():
    df = pd.DataFrame({"col1": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="No categorical data found in the input DataFrame."):
        extract_categorical(df)


def test_impute_numerical(numerical_df):
    df = numerical_df.copy()
    median = numerical_df["col3"].median()
    df.loc[0, "col3"] = np.nan  # introduce a NaN value

    imputed_df = impute_numerical(df)
    assert not imputed_df.isnull().any().any(), "Imputation failed, null values found"

    assert imputed_df.loc[1, "col3"] == median, "Imputation failed, expected median value"


def test_extract_numerical_columns(df, numerical_df):
    df = pd.concat([df, numerical_df], axis=1)

    extracted_df = extract_numerical(df)
    assert "col1" not in extracted_df.columns, "Non-numeric column found"
    assert "col2" not in extracted_df.columns, "Non-numeric column found"
    assert "col3" in extracted_df.columns, "Expected numeric column not found"
    assert "col4" in extracted_df.columns, "Expected numeric column not found"

    # Test with non-DataFrame input
    with pytest.raises(TypeError):
        extract_numerical("not a dataframe")

    # Test with empty DataFrame
    with pytest.raises(ValueError):
        extract_numerical(pd.DataFrame())

    # Test with DataFrame with no columns
    with pytest.raises(ValueError):
        extract_numerical(pd.DataFrame(index=[0, 1, 2]))

    # Test with DataFrame with no numeric columns
    with pytest.raises(ValueError):
        extract_numerical(pd.DataFrame({"col1": ["a", "b", "c"]}))
