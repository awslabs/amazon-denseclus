#!/usr/bin/env python3

"""
Utility functions for making fits to UMAP
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.datasets import make_blobs
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler


def extract_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts categorical features into binary dummy dataframe

    Parameters:
        df (pd.DataFrame): DataFrame with numerical and categorical features

    Returns:
        pd.DataFrame: binary dummy DataFrame of categorical features
    """

    categorical = df.select_dtypes(exclude=["float", "int"])
    if categorical.shape[1] == 0:
        raise ValueError("No Categories found, check that objects are in dataframe")

    categorical_dummies = pd.get_dummies(categorical)

    return categorical_dummies


def extract_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts numerical features into normalized numeric only dataframe

    Parameters:
        df (pd.DataFrame): DataFrame with numerical and categorical features

    Returns:
        pd.DataFrame: normalized numerical DataFrame of numerical features
    """

    numerical = df.select_dtypes(include=["float", "int"])
    if numerical.shape[1] == 0:
        raise ValueError("No numerics found, check that numerics are in dataframe")

    return transform_numerics(numerical)


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


def make_dataframe() -> pd.DataFrame:
    """This will create dataframe for demonstration purposes.

    Returns:
        pd.DataFrame: dataframe of categorical and numerical data
    """
    X, _ = make_blobs(n_samples=1000, n_features=8, random_state=10)  # ruff: noqa: W0632
    numerical = StandardScaler().fit_transform(X[:, :6])
    categorical = KBinsDiscretizer(n_bins=3, encode="ordinal").fit_transform(X[:, 6:])
    categorical = np.where(
        categorical == 1.0,
        "M",
        np.where(categorical == 2.0, "H", "L"),
    ).astype(str)

    numerical_columns = [f"num_{i}" for i in range(numerical.shape[1])]
    df = pd.DataFrame(numerical, columns=numerical_columns)

    categorical_columns = [f"cat_{i}" for i in range(categorical.shape[1])]
    for idx, c in enumerate(categorical_columns):
        df[c] = categorical[:, idx]

    return df
