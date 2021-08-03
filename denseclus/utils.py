#!/usr/bin/env python3
from warnings import filterwarnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer


def check_is_df(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Requires DataFrame as input")


def extract_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts categorical features into binary dummy dataframe

    Parameters:
        df (pd.DataFrame): DataFrame with numerical and categorical features

    Returns:
        pd.DataFrame: binary dummy DataFrame of categorical features
    """
    check_is_df(df)

    categorical = df.select_dtypes(exclude=["float", "int"])
    if categorical.shape[1] == 0:
        raise ValueError("No Categories found, check that objects are in dataframe")

    categorical_dummies = pd.get_dummies(categorical)

    return categorical_dummies


def extract_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts numerical features into normailzed numeric only dataframe

    Parameters:
        df (pd.DataFrame): DataFrame with numerical and categorical features

    Returns:
        pd.DataFrame: normalized numerical DataFrame of numerical features
    """
    check_is_df(df)

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

    check_is_df(numerical)

    for names in numerical.columns.tolist():

        pt = PowerTransformer(copy=False)
        # TO DO: fix this warning message
        filterwarnings("ignore")
        numerical.loc[:, names] = pt.fit_transform(
            np.array(numerical.loc[:, names]).reshape(-1, 1),
        )
        filterwarnings("default")

    return numerical
