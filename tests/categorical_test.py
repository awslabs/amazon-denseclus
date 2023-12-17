import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from denseclus.categorical import extract_categorical, impute_categorical


@pytest.mark.fast
def test_extract_categorical(categorical_df):
    categories = extract_categorical(categorical_df)
    assert categories.shape == (4, 3)
    assert categories.min().min() == 0
    assert categories.max().min() == 1


@pytest.mark.fast
def test_extract_categorical_is_df():
    with pytest.raises(TypeError):
        extract_categorical(["A", "B", "C"])


@pytest.mark.fast
def test_extract_categorical_is_object(numerical_df):
    with pytest.raises(ValueError):
        extract_categorical(numerical_df)


@pytest.mark.fast
def test_extract_categorical_empty_df():
    with pytest.raises(ValueError):
        extract_categorical(pd.DataFrame())


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
