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
from hdbscan import flat
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
            random_state : int, default=None
                Random State for both UMAP and numpy.random.
                If set to None UMAP will run in Numba in multicore mode but
                results may vary between runs.
                Setting a seed may help to offset the stochastic nature of
                UMAP by setting it with fixed random seed.

            n_neighbors : int, default=30
                Level of neighbors for UMAP.
                Setting this higher will generate higher densities at the expense
                of requiring more computational complexity.

            min_samples : int, default=15
                Samples used for HDBSCAN.
                The larger this is set the more noise points get declared and the
                more restricted clusters become to only dense areas.

            min_cluster_size : int, default=100
                Minimum Cluster size for HDBSCAN.
                The minimum number of points from which a cluster needs to be
                formed.

            n_components : int, default=logarithm
                Number of components for UMAP.
                These are dimensions to reduce the data down to.
                Ideally, this needs to be a value that preserves all the information
                to form meaningful clusters. Default is the logarithm of total
                number of features.

            cluster_selection_method: str, default=eom
                The HDBSCAN selection method for how flat clusters are selected from
                the cluster hierarchy. Defaults to EOM or Excess of Mass

            umap_combine_method : str, default=intersection
                Method by which to combine embeddings spaces.
                Options include: intersection, union, contrast,
                intersection_union_mapper
                The latter combines both the intersection and union of
                the embeddings.
                See:
                https://umap-learn.readthedocs.io/en/latest/composing_models.html

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

            flat_clusters: bool, default=False
                Instead of determining cluster size based on density,
                the algorithm will attempt to partition the data into the specified
                number of clusters and the resulting clusters will have a fixed size.

    """

    def __init__(
        self,
        random_state: int = 42,
        n_neighbors: int = 30,
        min_samples: int = 15,
        min_cluster_size: int = 100,
        n_components: int = 5,
        cluster_selection_method: str = "eom",
        umap_combine_method: str = "intersection",
        prediction_data: bool = False,
        verbose: bool = False,
        flat_clusters: bool = False,
    ):
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer")
        if not isinstance(min_samples, int) or min_samples <= 0:
            raise ValueError("min_samples must be a positive integer")
        if not isinstance(min_cluster_size, int) or min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be a positive integer")
        if umap_combine_method not in [
            "intersection",
            "union",
            "contrast",
            "intersection_union_mapper",
        ]:
            raise ValueError("umap_combine_method must be valid selection")

        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.n_components = n_components
        self.cluster_selection_method = cluster_selection_method
        self.umap_combine_method = umap_combine_method
        self.prediction_data = prediction_data
        self.flat_clusters = flat_clusters

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

        if not isinstance(self.n_components, int):
            self.n_components = int(round(np.log(df.shape[1])))

        logger.info("Extracting categorical features")
        self.categorical_ = extract_categorical(df)

        logger.info("Extracting numerical features")
        self.numerical_ = extract_numerical(df)

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
                metric="l2",
                n_neighbors=self.n_neighbors,
                n_components=self.n_components,
                min_dist=0.0,
                random_state=self.random_state,
                n_jobs=1 if self.random_state is not None else -1,
                verbose=False,
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
                metric="dice",
                n_neighbors=self.n_neighbors,
                n_components=self.n_components,
                min_dist=0.0,
                random_state=self.random_state,
                n_jobs=1 if self.random_state is not None else -1,
                verbose=False,
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
                n_neighbors=self.n_neighbors,
                n_components=self.n_components,
                min_dist=0.0,
                n_jobs=1 if self.random_state is not None else -1,
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
        # create clusters of a fixed size
        if self.flat_clusters:
            logger.info("Fitting HDBSCAN with flat clusters")
            flat_model_ = flat.HDBSCAN_flat(
                X=self.mapper_.embedding_,
                cluster_selection_method=self.cluster_selection_method,
                n_clusters=self.flat_clusters,
                min_samples=self.min_samples,
                metric="euclidean",
            )

            self.hdbscan_ = flat_model_
        # or find the ideal number of clusters based on the density
        else:
            logger.info("Fitting HDBSCAN with default parameters")
            hdb_ = hdbscan.HDBSCAN(
                min_samples=self.min_samples,
                min_cluster_size=self.min_cluster_size,
                cluster_selection_method=self.cluster_selection_method,
                prediction_data=self.prediction_data,
                gen_min_span_tree=True,
                metric="euclidean",
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
