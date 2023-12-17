import numpy as np
import pandas as pd
import pytest

from denseclus.numerical import (
    extract_numerical,
    transform_numerics,
    impute_numerical,
)


@pytest.mark.fast
def test_extract_numerical(numerical_df):
    numerics = extract_numerical(numerical_df)
    assert str(numerics["col3"].dtypes) == "float64"
    assert str(numerics["col4"].dtypes) == "float64"


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
def test_extract_numerical_empty_df():
    with pytest.raises(ValueError):
        extract_numerical(pd.DataFrame())


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

    with pytest.raises(TypeError):
        extract_numerical("not a dataframe")

    with pytest.raises(ValueError):
        extract_numerical(pd.DataFrame())

    with pytest.raises(ValueError):
        extract_numerical(pd.DataFrame(index=[0, 1, 2]))

    with pytest.raises(ValueError):
        extract_numerical(pd.DataFrame({"col1": ["a", "b", "c"]}))


@pytest.mark.fast
def test_impute_numerical_strategy_error(numerical_df):
    with pytest.raises(ValueError):
        impute_numerical(numerical_df, strategy="constant")


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
