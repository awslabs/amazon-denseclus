"""
Utility functions for making fits to UMAP
"""
import os
import random

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler


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


# def import_gpu_libraries() -> dict:
#     """
#     Helper function to import the cuml and cuml.manifold.umap libraries

#     Returns:
#         dict : Configuration for cuml umap and hdscan
#     """
#     gpu_libs = {"cuml": None, "cuml.manifold.umap": None, "cuml.cluster": None}
#     for lib, _ in gpu_libs.items():
#         if find_spec(lib):
#             gpu_libs[lib] = importlib.import_module(lib)
#     return gpu_libs
