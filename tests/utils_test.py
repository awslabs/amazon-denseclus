#!/usr/bin/env python3
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


@pytest.mark.fast
def test_extract_categorical(categorical_df):
    categories = extract_categorical(categorical_df)
    assert categories.shape == (4, 3)
    assert categories.min().min() == 0
    assert categories.max().min() == 1


@pytest.mark.fast
def test_extract_numerical(numerical_df):
    numerics = extract_numerical(numerical_df)
    assert str(numerics["col3"].dtypes) == "float64"
    assert str(numerics["col4"].dtypes) == "float64"


@pytest.mark.fast
def test_extract_categorical_is_df():
    with pytest.raises(TypeError):
        extract_categorical(["A", "B", "C"])


@pytest.mark.fast
def test_extract_categorical_is_object(numerical_df):
    with pytest.raises(ValueError):
        extract_categorical(numerical_df)


@pytest.mark.fast
def test_extract_numerical_is_df():
    with pytest.raises(TypeError):
        extract_numerical([1, 2, 3])


@pytest.mark.fast
def test_extract_numerical_is_numeric(categorical_df):
    with pytest.raises(ValueError):
        extract_numerical(categorical_df)


@pytest.mark.fast
def test_transform_numerics(numerical_df):
    numerics = transform_numerics(numerical_df)
    assert len(numerics) == 3


@pytest.mark.fast
def test_transform_numerics_is_df():
    with pytest.raises(AttributeError):
        transform_numerics([1, 2, 3])


@pytest.mark.fast
def test_extract_categorical_empty_df():
    with pytest.raises(ValueError):
        extract_categorical(pd.DataFrame())


@pytest.mark.fast
def test_extract_numerical_empty_df():
    with pytest.raises(ValueError):
        extract_numerical(pd.DataFrame())


@pytest.mark.fast
def test_extract_categorical_no_categorical(categorical_df):
    df = categorical_df.select_dtypes(include=["float", "int"])
    with pytest.raises(ValueError):
        extract_categorical(df)


@pytest.mark.fast
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


@pytest.mark.fast
def test_extract_categorical_with_high_cardinality(categorical_df):
    df = pd.DataFrame(
        {"col1": ["A", "B", "A", "B"], "col2": ["C", "D", "E", "C"], "col3": ["A", "A", "B", "B"]},
    )
    result = extract_categorical(df, cardinality_threshold=1)
    assert result.sum(axis=0).sum() == 0
    assert result.sum(axis=1).max(axis=0) == 1.0
    assert result.shape == (4, 3)


@pytest.mark.fast
def test_impute_categorical_with_missing_values(missing_categorical_df):
    expected = pd.DataFrame({"A": ["alpha", "Missing", "beta"], "B": ["gamma", "delta", "Missing"]})
    result = impute_categorical(missing_categorical_df, strategy="constant", fill_value="Missing")
    assert_frame_equal(result, expected)


@pytest.mark.skip(reason="expiremental feature")
def test_impute_categorical_with_custom_strategy(missing_categorical_df):
    expected = pd.DataFrame({"A": ["alpha", "Z", "beta"], "B": ["gamma", "delta", "Z"]})
    result = impute_categorical(
        missing_categorical_df,
        strategy="constant",
        fill_value="Missing",
        custom_strategy=lambda s: "Z",
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.fast
def test_extract_categorical_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame should not be empty."):
        extract_categorical(df)


@pytest.mark.fast
def test_extract_categorical_no_columns():
    df = pd.DataFrame(index=[0, 1, 2])
    with pytest.raises(ValueError, match="Input DataFrame should not be empty."):
        extract_categorical(df)


@pytest.mark.fast
def test_extract_categorical_no_categorical_data():
    df = pd.DataFrame({"col1": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="No categorical data found in the input DataFrame."):
        extract_categorical(df)


@pytest.mark.fast
def test_impute_numerical(numerical_df):
    df = numerical_df.copy()
    median = numerical_df["col3"].median()
    df.loc[0, "col3"] = np.nan  # introduce a NaN value

    imputed_df = impute_numerical(df)
    assert not imputed_df.isnull().any().any(), "Imputation failed, null values found"

    assert imputed_df.loc[1, "col3"] == median, "Imputation failed, expected median value"


@pytest.mark.fast
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


@pytest.mark.fast
def test_impute_categorical_strategy_error(categorical_df):
    with pytest.raises(ValueError):
        impute_categorical(categorical_df, strategy="median", fill_value="Error")


@pytest.mark.fast
def test_impute_numerical_strategy_error(numerical_df):
    with pytest.raises(ValueError):
        impute_numerical(numerical_df, strategy="constant")


@pytest.mark.fast
def test_impute_categorical_constant(missing_categorical_df):
    result = impute_categorical(missing_categorical_df, "constant", "Missing")
    expected = pd.DataFrame({"A": ["alpha", "Missing", "beta"], "B": ["gamma", "delta", "Missing"]})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.fast
def test_impute_categorical_most_frequent():
    df = pd.DataFrame({"A": ["alpha", np.nan, "alpha"], "B": ["gamma", "gamma", np.nan]})
    result = impute_categorical(df, "most_frequent", "Missing")
    expected = pd.DataFrame({"A": ["alpha", "alpha", "alpha"], "B": ["gamma", "gamma", "gamma"]})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.skip(reason="expiremental feature")
def test_impute_categorical_custom_strategy(missing_categorical_df):
    def second_most_frequent(s):
        return s.value_counts().index[1] if len(s.value_counts()) > 1 else s.value_counts().index[0]

    result = impute_categorical(
        missing_categorical_df,
        "constant",
        "Missing",
        custom_strategy=second_most_frequent,
    )
    expected = pd.DataFrame({"A": ["alpha", "alpha", "beta"], "B": ["gamma", "delta", "delta"]})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.fast
def test_impute_categorical_no_missing_values():
    df = pd.DataFrame({"A": ["alpha", "beta", "alpha"], "B": ["gamma", "delta", "gamma"]})
    result = impute_categorical(df, "constant", "Missing")
    pd.testing.assert_frame_equal(result, df)


@pytest.mark.fast
def test_impute_categorical_invalid_strategy(missing_categorical_df):
    with pytest.raises(ValueError):
        impute_categorical(missing_categorical_df, "invalid", "Missing")


@pytest.mark.fast
def test_impute_numerical_median(missing_numerical_df):
    result = impute_numerical(missing_numerical_df, "median")
    expected = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5, 4.5]})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.fast
def test_impute_numerical_mean(missing_numerical_df):
    result = impute_numerical(missing_numerical_df, "mean")
    expected = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5, 4.5]})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.fast
def test_impute_numerical_no_missing_values():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = impute_numerical(df, "median")
    pd.testing.assert_frame_equal(result, df)


@pytest.mark.fast
def test_impute_numerical_invalid_strategy(missing_numerical_df):
    with pytest.raises(ValueError):
        impute_numerical(missing_numerical_df, "invalid")
