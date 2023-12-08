"""
Functions for handling numeric values
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer


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
