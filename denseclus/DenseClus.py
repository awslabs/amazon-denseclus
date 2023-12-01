"""
This file contains the implementation of DenseClus.

DenseClus provides a unified interface for clustering mixed data types.
It supports various methods for combining the embeddings:
   including 'intersection', 'union', 'contrast', and 'intersection_union_mapper'.

:Authors: Charles Frenzel, Baichaun Sun
:Date: November 2023

Usage:
   .. doctest::
   # Create a DenseClus object
   dense_clus = DenseClus(
       umap_combine_method="intersection_union_mapper",
   )

   # Fit the DenseClus object to your mixed data type dataframe
   dense_clus.fit(df)

   # Return the clusters from the dataframe
   clusters = dense_clus.score()
"""


import logging
import warnings

import hdbscan
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.base import BaseEstimator, ClassifierMixin
from copy import deepcopy
from .utils import extract_categorical, extract_numerical

logger = logging.getLogger("denseclus")
logger.setLevel(logging.ERROR)
sh = logging.StreamHandler()
sh.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)
logger.addHandler(sh)

# this suppresses the dice metric warning
warnings.filterwarnings("ignore", category=UserWarning)


class DenseClus(BaseEstimator, ClassifierMixin):
    """
    DenseClus

    Creates UMAP embeddings and HDSCAN clusters from mixed data

    :param random_state: Random State for both UMAP and numpy.random.
    If set to None UMAP will run in Numba in multicore mode but
    results may vary between runs.
    Setting a seed may help to offset the stochastic nature of
    UMAP by setting it with fixed random seed.
    :type random_state: int, default=42

    :param umap_combine_method: Method by which to combine embeddings spaces.
    Options include: intersection, union, contrast,
    methods for combining the embeddings: including
    'intersection', 'union', 'contrast', and 'intersection_union_mapper'.
    :type umap_combine_method: str, default=intersection

    :param verbose: Level of verbosity to print when fitting and predicting.
    Setting to False will only show Warnings that appear.
    :type verbose: bool, default=False

    :param umap_params: A dictionary containing dictionaries: 'categorical', 'numerical' and 'combined' if
    'intersection_union_mapper' is selected as the 'umap_combine_method'.
    Each dictionary should contain parameters for the UMAP algorithm used to
    fit the data.
    If not provided, default UMAP parameters will be used.
    :type umap_params: dict, optional

    :param hdbscan_params: A dictionary containing parameters for the HDBSCAN algorithm.
    If not provided, default HDBSCAN parameters will be used.
    :type hdbscan_params: dict, optional

    """

    def __init__(
        self,
        random_state: int = 42,
        umap_combine_method: str = "intersection",
        verbose: bool = False,
        umap_params=None,
        hdbscan_params=None,
        **kwargs,
    ):
        if umap_combine_method not in [
            "intersection",
            "union",
            "contrast",
            "intersection_union_mapper",
        ]:
            raise ValueError("umap_combine_method must be valid selection")

        self.random_state = random_state
        self.umap_combine_method = umap_combine_method

        # Default parameters
        default_umap_params = {
            "categorical": {
                "metric": "dice",
                "n_neighbors": 30,
                "n_components": 5,
                "min_dist": 0.0,
            },
            "numerical": {
                "metric": "l2",
                "n_neighbors": 30,
                "n_components": 5,
                "min_dist": 0.0,
            },
            "combined": {
                "n_neighbors": 30,
                "min_dist": 0.0,
                "n_components": 5,
            },
        }

        default_hdbscan_params = {
            "min_cluster_size": 100,
            "min_samples": 15,
            "gen_min_span_tree": True,
            "metric": "euclidean",
        }

        if umap_params:
            for key in umap_params:
                if key in default_umap_params:
                    default_umap_params[key].update(umap_params[key])  # type: ignore # noqa
                else:
                    raise ValueError(f"Invalid key '{key}' in umap_params")
            self.umap_params = default_umap_params
        else:
            self.umap_params = default_umap_params

        if hdbscan_params:
            self.hdbscan_params = hdbscan_params  # pragma: no cover
        else:
            self.hdbscan_params = default_hdbscan_params

        if verbose:  # pragma: no cover
            logger.setLevel(logging.DEBUG)
            self.verbose = True
        else:  # pragma: no cover
            logger.setLevel(logging.ERROR)
            self.verbose = False

            # suppress deprecation warnings
            # see: https://stackoverflow.com/questions/54379418
            # pylint: disable=W0613
            def noop(*args, **kargs):
                pass

            warnings.warn = noop

        if isinstance(random_state, int):
            np.random.seed(seed=random_state)
        else:  # pragma: no cover
            logger.info("No random seed passed, running UMAP in Numba, parallel")

        self.kwargs = kwargs

    def __repr__(self):  # pragma: no cover
        return f"""DenseClus(random_state={self.random_state}
                            ,umap_combine_method={self.umap_combine_method}
                            ,umap_params={self.umap_params}
                            ,hdbscan_params={self.hdbscan_params})"""

    def __str__(self):  # pragma: no cover
        return f"""DenseClus object (random_state={self.random_state}
                                    ,umap_combine_method={self.umap_combine_method}
                                    ,verbose={self.verbose})"""

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fits the UMAP and HDBSCAN models to the provided data.

        :param df: DataFrame object with named columns of categorical and numerics
        :type df: pandas DataFrame
        :return: Fitted UMAPs and HDBSCAN
        :rtype: None
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Requires DataFrame as input")

        logger.info("Extracting categorical features")
        self.categorical_ = extract_categorical(df, **self.kwargs)

        logger.info("Extracting numerical features")
        self.numerical_ = extract_numerical(df, **self.kwargs)

        logger.info("Fitting categorical UMAP")
        self._fit_categorical()

        logger.info("Fitting numerical UMAP")
        self._fit_numerical()

        logger.info("Mapping/Combining Embeddings")
        self.mapper_ = self._umap_embeddings()

        logger.info("Fitting HDBSCAN...")
        self._fit_hdbscan()

    def _fit_numerical(self) -> None:
        """
        Fits UMAP based on numerical data

        :return: fitted numerical umap
        :rtype: None
        """
        try:
            logger.info("Fitting UMAP for Numerical data")

            numerical_umap = umap.UMAP(
                random_state=self.random_state,
                n_jobs=1 if self.random_state is not None else -1,
                verbose=False,
                **self.umap_params["numerical"],
                low_memory=False,
            ).fit(self.numerical_)

            self.numerical_umap_ = numerical_umap
            logger.info("Numerical UMAP fitted successfully")

        except Exception as e:
            logger.error("Failed to fit numerical UMAP: %s", str(e))
            raise

    def fit_predict(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> np.array:
        """
        Generate predictions on new data points.
        Refits UMAP embeddings and then uses the HDBSCAN's approximate_predict function to predict the cluster labels and strengths.

        :param df_old: The old data for which to generate predictions. This should be a DataFrame with the same structure as the one used in the fit method.
        :type df_old: pd.DataFrame
        :param df_new: The new data for which to generate predictions. This should be a DataFrame with the same structure as the one used in the fit method.
        :type df_new: pd.DataFrame
        :return: The predicted cluster labels for each row in df_new.
        :rtype: np.array
        :return: The strengths of the predictions for each row in df_new.
        :rtype: np.array
        """
        df_old_len = len(df_old)
        df_combined = pd.concat([df_old, df_new])

        categorical_combined = deepcopy(extract_categorical(df_combined, **self.kwargs))
        numerical_combined = deepcopy(extract_numerical(df_combined, **self.kwargs))

        self.categorical_umap_ = deepcopy(self.categorical_umap_.fit(categorical_combined))  # type: ignore
        self.numerical_umap_ = deepcopy(self.numerical_umap_.fit(numerical_combined))

        # Perform the operations on the new embeddings
        mapper = self._umap_embeddings()

        labels, strengths = hdbscan.approximate_predict(
            self.hdbscan_,
            mapper.embedding_[df_old_len:, :],
        )

        return np.stack((labels, strengths), axis=-1)

    def _fit_categorical(self) -> None:
        """
        Fit a UMAP based on categorical data

        :return: fitted categorical umap
        :rtype: None
        """
        try:
            logger.info("Fitting UMAP for categorical data")

            categorical_umap = umap.UMAP(
                random_state=self.random_state,
                n_jobs=1 if self.random_state is not None else -1,
                verbose=False,
                low_memory=False,
                **self.umap_params["categorical"],
            ).fit(self.categorical_)
            self.categorical_umap_ = categorical_umap
            logger.info("Categorical UMAP fitted successfully")

        except Exception as e:
            logger.error("Failed to fit numerical UMAP: %s", str(e))
            raise

    def _umap_embeddings(self) -> "umap":
        """Combines the numerical and categorical UMAP embeddings based on the specified method.

        Supported: 'intersection', 'union', 'contrast', and 'intersection_union_mapper'

        :return: the combined UMAPs
        :rtype: UMAP
        """
        logger.info("Combining UMAP embeddings using method: %s", self.umap_combine_method)
        if self.umap_combine_method == "intersection":
            mapper = deepcopy(self.numerical_umap_ * self.categorical_umap_)

        elif self.umap_combine_method == "union":
            mapper = deepcopy(self.numerical_umap_ + self.categorical_umap_)

        elif self.umap_combine_method == "contrast":
            mapper = deepcopy(self.numerical_umap_ - self.categorical_umap_)

        elif self.umap_combine_method == "intersection_union_mapper":
            intersection_mapper = umap.UMAP(
                random_state=self.random_state,
                n_jobs=1 if self.random_state is not None else -1,
                low_memory=False,
                **self.umap_params["combined"],
            ).fit(self.numerical_)
            mapper = deepcopy(intersection_mapper * (self.numerical_umap_ + self.categorical_umap_))

        else:
            logger.error("Invalid UMAP combine method: %s", self.umap_combine_method)
            raise ValueError("Select valid UMAP combine method")

        return mapper

    def _fit_hdbscan(self) -> None:
        """
        Fits HDBSCAN to the combined embeddings.

        :return: fitted hdbscan
        :rtype: None
        """
        logger.info("Fitting HDBSCAN with default parameters")
        hdb_ = hdbscan.HDBSCAN(
            prediction_data=True,
            **self.hdbscan_params,
        ).fit(self.mapper_.embedding_)
        self.hdbscan_ = hdb_

        logger.info("HDBSCAN fit")

    def score(self) -> np.array:
        """
        Returns the cluster assigned to each row.

        This is wrapper function for HDBSCAN. It outputs the cluster labels
        that HDBSCAN converged on.

        :return: labels
        :rtype: np.array([int])
        """
        return self.hdbscan_.labels_
