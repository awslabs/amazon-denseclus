#!/usr/bin/env python3

"""
This file contains the implementation of DenseClus.

DenseClus provides a unified interface for clustering mixed data types.
It supports various methods for combining the embeddings:
    including 'intersection', 'union', 'contrast', and 'intersection_union_mapper'.

Usage:
    # Create a DenseClus object
    dense_clus = DenseClus(
        umap_combine_method="intersection_union_mapper",
    )

    # Fit the DenseClus object to your mixed data type dataframe
    dense_clus.fit(df)

    # Return the clusters from the dataframe
    clusters = dense_clus.score()


Authors: Charles Frenzel, Baichaun Sun
Date: November 2023
"""


import logging
import warnings

import hdbscan
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.base import BaseEstimator, ClassifierMixin

from .utils import extract_categorical, extract_numerical

logger = logging.getLogger("denseclus")
logger.setLevel(logging.ERROR)
sh = logging.StreamHandler()
sh.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)
logger.addHandler(sh)


class DenseClus(BaseEstimator, ClassifierMixin):
    """DenseClus

    Creates UMAP embeddings and HDSCAN clusters from mixed data

    Parameters
    ----------
            random_state : int, default=42
                Random State for both UMAP and numpy.random.
                If set to None UMAP will run in Numba in multicore mode but
                results may vary between runs.
                Setting a seed may help to offset the stochastic nature of
                UMAP by setting it with fixed random seed.

            umap_combine_method : str, default=contrast
                Method by which to combine embeddings spaces.
                Options include: intersection, union, contrast,
                methods for combining the embeddings: including
                'intersection', 'union', 'contrast', and 'intersection_union_mapper'.

                'intersection' preserves the numerical embeddings more, focusing on the quantitative aspects of
                the data. This method is particularly useful when the numerical data is of higher importance or
                relevance to the clustering task.

                'Union' preserves the categorical embeddings more, emphasizing the qualitative aspects of the
                data. This method is ideal when the categorical data carries significant weight or importance in
                the clustering task.

                'Contrast' highlights the differences between the numerical and categorical embeddings, providing
                a more balanced representation of both. This method is particularly useful when there are
                significant differences between the numerical and categorical data, and both types of data are
                equally important for the clustering task.

                'Intersection_union_mapper' is a hybrid method that combines the strengths of both 'intersection'
                and 'union'. It first applies the 'intersection' method to preserve the numerical embeddings, then
                applies the 'union' method to preserve the categorical embeddings. This method is useful when both
                numerical and categorical data are important, but one type of data is not necessarily more
                important than the other.
                See: https://umap-learn.readthedocs.io/en/latest/composing_models.html

            prediction_data: bool, default=False
                Whether to generate extra cached data for predicting labels or
                membership vectors few new unseen points later. If you wish to
                persist the clustering object for later reuse you probably want
                to set this to True.
                See:
                https://hdbscan.readthedocs.io/en/latest/soft_clustering.html

            verbose : bool, default=False
                Level of verbosity to print when fitting and predicting.
                Setting to False will only show Warnings that appear.

            umap_params : dict, optional
                A dictionary containing dictionaries: 'categorical', 'numerical' and 'combined' if
                'intersection_union_mapper' is selected as the 'umap_combine_method'.
                Each dictionary should contain parameters for the UMAP algorithm used to
                fit the data.
                If not provided, default UMAP parameters will be used.

                Example:
                umap_params = {
                                'categorical': {'n_neighbors': 15, 'min_dist': 0.1},
                                'numerical': {'n_neighbors': 20, 'min_dist': 0.2}
                                'combined' : {'n_neighbors': 5, 'min_dist': 0.1}
                            }

            hdbscan_params : dict, optional
                A dictionary containing parameters for the HDBSCAN algorithm.
                If not provided, default HDBSCAN parameters will be used.

                Example:
                hdbscan_params = {'min_cluster_size': 10}
    """

    def __init__(
        self,
        random_state: int = 42,
        umap_combine_method: str = "contrast",
        prediction_data: bool = False,
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
        self.prediction_data = prediction_data

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

        # self.umap_params = dict()
        # if umap_params is None:
        #     self.umap_params = default_umap_params
        # else:
        #     for key, new_params in umap_params.items():
        #         self.umap_params[key] = default_umap_params | new_params
        if umap_params is None:
            self.umap_params = default_umap_params
        else:
            for key in umap_params:
                if key in default_umap_params:
                    default_umap_params[key].update(umap_params[key])  # type: ignore # noqa
                else:
                    raise ValueError(f"Invalid key '{key}' in umap_params")
            self.umap_params = default_umap_params

        if hdbscan_params is None:
            self.hdbscan_params = default_hdbscan_params
        else:
            default_hdbscan_params.update(hdbscan_params)
            self.hdbscan_params = default_hdbscan_params

        if verbose:
            logger.setLevel(logging.DEBUG)
            self.verbose = True
        else:
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
        else:
            logger.info("No random seed passed, running UMAP in Numba, parallel")

        self.kwargs = kwargs

    def __repr__(self):
        return str(self.__dict__)

    def fit(self, df: pd.DataFrame) -> None:
        """Fits the UMAP and HDBSCAN models to the provided data.

        Parameters
        ----------
            df : pandas DataFrame
                DataFrame object with named columns of categorical and numerics

        Returns
        -------
            Fitted: None
                Fitted UMAPs and HDBSCAN
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
        self._umap_embeddings()

        logger.info("Fitting HDBSCAN...")
        self._fit_hdbscan()

    def _fit_numerical(self):
        """
        Fit a UMAP based on numerical data

        Returns:
            self
        """
        try:
            logger.info("Fitting UMAP for Numerical data")

            numerical_umap = umap.UMAP(
                random_state=self.random_state,
                n_jobs=1 if self.random_state is not None else -1,
                verbose=False,
                **self.umap_params["numerical"],
            ).fit(self.numerical_)

            self.numerical_umap_ = numerical_umap
            logger.info("Numerical UMAP fitted successfully")

            return self

        except Exception as e:
            logger.error("Failed to fit numerical UMAP: %s", str(e))
            raise

    def _fit_categorical(self):
        """
        Fit a UMAP based on categorical data

        Returns:
            self
        """
        try:
            logger.info("Fitting UMAP for categorical data")

            categorical_umap = umap.UMAP(
                random_state=self.random_state,
                n_jobs=1 if self.random_state is not None else -1,
                verbose=False,
                **self.umap_params["categorical"],
            ).fit(self.categorical_)
            self.categorical_umap_ = categorical_umap
            logger.info("Categorical UMAP fitted successfully")
            return self

        except Exception as e:
            logger.error("Failed to fit numerical UMAP: %s", str(e))
            raise

    def _umap_embeddings(self):
        """Combines the numerical and categorical UMAP embeddings based on the specified method.

        Supported: 'intersection', 'union', 'contrast', and 'intersection_union_mapper'

        Returns
        -------
            self
        """
        logger.info("Combining UMAP embeddings using method: %s", self.umap_combine_method)
        if self.umap_combine_method == "intersection":
            self.mapper_ = self.numerical_umap_ * self.categorical_umap_

        elif self.umap_combine_method == "union":
            self.mapper_ = self.numerical_umap_ + self.categorical_umap_

        elif self.umap_combine_method == "contrast":
            self.mapper_ = self.numerical_umap_ - self.categorical_umap_

        elif self.umap_combine_method == "intersection_union_mapper":
            intersection_mapper = umap.UMAP(
                random_state=self.random_state,
                n_jobs=1 if self.random_state is not None else -1,
                **self.umap_params["combined"],
            ).fit(self.numerical_)
            self.mapper_ = intersection_mapper * (self.numerical_umap_ + self.categorical_umap_)

        else:
            logger.error("Invalid UMAP combine method: %s", self.umap_combine_method)
            raise ValueError("Select valid UMAP combine method")

        return self

    def _fit_hdbscan(self):
        """Fits HDBSCAN to the combined embeddings.

        Parameters
        ----------
            None : None
        Returns
        -------
            self
        """
        logger.info("Fitting HDBSCAN with default parameters")
        hdb_ = hdbscan.HDBSCAN(
            prediction_data=self.prediction_data,
            **self.hdbscan_params,
        ).fit(self.mapper_.embedding_)
        self.hdbscan_ = hdb_

        logger.info("HDBSCAN fit")
        return self

    def score(self):
        """Returns the cluster assigned to each row.

        This is wrapper function for HDBSCAN. It outputs the cluster labels
        that HDBSCAN converged on.

        Parameters
        ----------
        None : None

        Returns
        -------
        labels : np.array([int])
        """
        return self.hdbscan_.labels_
