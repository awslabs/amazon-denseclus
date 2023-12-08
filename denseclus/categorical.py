"""
Functions for handling categorical values
"""

from typing import Callable, Optional

import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer


def extract_categorical(
    df: pd.DataFrame,
    cardinality_threshold: int = 25,
    strategy: str = "constant",
    fill_value="Missing",
    **kwargs,
) -> pd.DataFrame:
    """Extracts categorical features into binary dummy dataframe

    Parameters:
        df (pd.DataFrame): DataFrame with numerical and categorical features
        cardinality_threshold: (int): Threshold to revert to using hashing when the number of
        categorical features are high. Default: 25
        **kwargs : Additional arguments to pass to imputation, allows to customize
            Note: Imputation defaults to filling with 'Missing'

    Returns:
        pd.DataFrame: binary dummy DataFrame of categorical features
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input should be a pandas DataFrame")
    if df.empty:
        raise ValueError("Input DataFrame should not be empty.")

    categorical = df.select_dtypes(exclude=["float", "int"])

    if categorical.empty:
        raise ValueError("No categorical data found in the input DataFrame.")

    categorical = impute_categorical(
        categorical,
        strategy=strategy,
        fill_value=fill_value,
        **kwargs,
    )

    max_cardinality = max(categorical.nunique())

    if max_cardinality > cardinality_threshold:
        print(f"Max of {max_cardinality} is greater than threshold {cardinality_threshold}")
        print("Hashing categorical features")
        hasher = FeatureHasher(n_features=cardinality_threshold, input_type="string")
        hashed_df = pd.DataFrame()
        for col in categorical.columns:
            hashed_features = hasher.transform(categorical[col].apply(lambda x: [x]))
            hashed_features = pd.DataFrame(hashed_features.toarray())
            hashed_df = pd.concat([hashed_df, hashed_features], axis=1)

        categorical = hashed_df
    else:
        categorical = pd.get_dummies(categorical, drop_first=True)

    return categorical


def impute_categorical(
    categorical: pd.DataFrame,
    strategy: str,
    fill_value: str,
    custom_strategy: Optional[Callable[[pd.Series], object]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Imputes missing values in categorical features.

    Parameters:
        df (pd.DataFrame): DataFrame with categorical features
        strategy (str, optional): The imputation strategy. Default is 'constant'.
        fill_value (str, optional): The value to use for imputation when strategy is 'constant'. Default is 'Missing'.
        custom_strategy (callable, optional): A custom function to compute the imputation value. Should take a Series and return an object.

    Returns:
        pd.DataFrame: DataFrame with imputed categorical features

    Example:
        To use a custom strategy that imputes missing values with the second most frequent value, you can do:

        def second_most_frequent(s):
            return s.value_counts().index[1] if len(s.value_counts()) > 1 else s.value_counts().index[0]

        impute_categorical(df, custom_strategy=second_most_frequent)
    """
    if strategy not in ["constant", "most_frequent"]:
        raise ValueError(f"Invalid strategy for categorical: {strategy}")

    if categorical.isnull().sum().sum() == 0:
        return categorical

    for col in categorical.columns:
        if custom_strategy:
            fill_value = custom_strategy(categorical[col])  # type: ignore
            categorical[col].fillna(fill_value, inplace=True)
        else:
            imputer = SimpleImputer(strategy=strategy, fill_value=fill_value, **kwargs)
            categorical[col] = imputer.fit_transform(categorical[[col]])[:, 0]

    return categorical
