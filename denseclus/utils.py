#!/usr/bin/env python3

"""
Utility functions for making fits to UMAP
"""
import warnings
from typing import Callable, Optional
import random
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, PowerTransformer, StandardScaler
from sklearn.datasets import make_classification


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


def extract_numerical(df: pd.DataFrame, impute_strategy: str = "median", **kwargs) -> pd.DataFrame:
    """Extracts numerical features into normalized numeric only dataframe

    Parameters:
        df (pd.DataFrame): DataFrame with numerical and categorical features
        impute_strategy (str): The imputation strategy to use if null values are found. Default is 'median'

    Returns:
        pd.DataFrame: normalized numerical DataFrame of numerical features
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input should be a pandas DataFrame")
    if df.empty:
        raise ValueError("Input DataFrame should not be empty.")

    numerical = df.select_dtypes(include=["float", "int"])
    if numerical.shape[1] == 0:
        raise ValueError("No numerics found, check that numerics are in dataframe")

    numerical = impute_numerical(numerical, strategy=impute_strategy, **kwargs)

    return transform_numerics(numerical)


def impute_numerical(numerical: pd.DataFrame, strategy: str = "median", **kwargs) -> pd.DataFrame:
    """Imputes numerical features with the given strategy

    Parameters:
        numerical (pd.DataFrame): DataFrame with numerical features
        strategy (str): The imputation strategy. Default is 'median'

    Returns:
        pd.DataFrame: DataFrame with imputed numerical features
    """
    if strategy not in ["median", "mean"]:
        raise ValueError(f"Invalid strategy for numerical: {strategy}")

    if numerical.isnull().sum().sum() == 0:
        return numerical

    imputer = SimpleImputer(strategy=strategy, **kwargs)
    numerical_imputed = pd.DataFrame(imputer.fit_transform(numerical), columns=numerical.columns)

    return numerical_imputed


def transform_numerics(numerical: pd.DataFrame) -> pd.DataFrame:
    """Power transforms numerical DataFrame

    Parameters:
        numerical (pd.DataFrame): Numerical features DataFrame

    Returns:
        pd.DataFrame: Normalized DataFrame of Numerical features
    """

    for name in numerical.columns.tolist():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            pt = PowerTransformer(copy=False)
            numerical[name] = pt.fit_transform(np.array(numerical[name]).reshape(-1, 1))

    return numerical


def make_dataframe(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """This will create dataframe for demonstration purposes.

    Returns:
        pd.DataFrame: dataframe of categorical and numerical data
    """
    X, _ = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=8,
        random_state=random_state,
    )
    numerical = StandardScaler().fit_transform(X[:, :6])
    categorical = KBinsDiscretizer(n_bins=5, encode="ordinal").fit_transform(X[:, 6:])
    categorical = np.where(
        categorical == 1.0,
        "M",
        np.where(
            categorical == 2.0,
            "H",
            np.where(categorical == 3.0, "MH", np.where(categorical == 4.0, "HL", "L")),
        ),
    ).astype(str)

    numerical_columns = [f"num_{i}" for i in range(numerical.shape[1])]
    df = pd.DataFrame(numerical, columns=numerical_columns)

    categorical_columns = [f"cat_{i}" for i in range(categorical.shape[1])]
    for idx, c in enumerate(categorical_columns):
        df[c] = categorical[:, idx]

    return df


def seed_everything(seed: int = 42):
    """
    Helper function to sett the random seed for everything to get better
    reproduction of results
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
