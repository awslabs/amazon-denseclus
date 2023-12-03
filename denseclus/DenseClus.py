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

from .utils import extract_categorical, extract_numerical, seed_everything

logger = logging.getLogger("denseclus")
logger.setLevel(logging.ERROR)
sh = logging.StreamHandler()
sh.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)
logger.addHandler(sh)

# this suppresses the jaccard metric warning(s)
warnings.filterwarnings("ignore", category=UserWarning)


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

            umap_combine_method : str, default=intersection
                Method by which to combine embeddings spaces.
                Options include: intersection, union, contrast,
                methods for combining the embeddings: including
                'intersection', 'union', 'contrast','intersection_union_mapper', and 'ensemble'

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

                'Ensemble' does not combine umaps at all and instead will keep them separate with ability to
                run preidction on new points using approximate_predict from HDSCAN. Points are voted on from both
                emedding layers with ties being broken via assignment probabilities.

                See: https://umap-learn.readthedocs.io/en/latest/composing_models.html

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
            "ensemble",
        ]:
            raise ValueError("umap_combine_method must be valid selection")

        self.random_state = random_state
        seed_everything(self.random_state)

        self.umap_combine_method = umap_combine_method

        # Default parameters
        default_umap_params = {
            "categorical": {
                "metric": "jaccard",
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
        if self.umap_combine_method != "ensemble":
            self._umap_embeddings()

        logger.info("Fitting HDBSCAN...")
        self._fit_hdbscan()

    def _fit_numerical(self) -> None:
        """
        Fit a UMAP based on numerical data

        Returns:
            None
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

        except Exception as e:
            logger.error("Failed to fit numerical UMAP: %s", str(e))
            raise

    def _fit_categorical(self) -> None:
        """
        Fit a UMAP based on categorical data

        Returns:
            None
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

        except Exception as e:
            logger.error("Failed to fit numerical UMAP: %s", str(e))
            raise

    def _umap_embeddings(self) -> None:
        """Combines the numerical and categorical UMAP embeddings based on the specified method.

        Supported: 'intersection', 'union', 'contrast', and 'intersection_union_mapper'

        Returns
        -------
            None
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

    def _fit_hdbscan(self) -> None:
        """Fits HDBSCAN to the combined embeddings.

        Parameters
        ----------
            None : None
        Returns
        -------
            None
        """
        logger.info("Fitting HDBSCAN")
        if self.umap_combine_method != "ensemble":
            logger.info("Fitting single HDBSCAN")
            hdb_ = hdbscan.HDBSCAN(
                **self.hdbscan_params,
            ).fit(self.mapper_.embedding_)
            self.hdbscan_ = hdb_
            self.labels_ = hdb_.labels_
            self.probabilities_ = hdb_.probabilities_

        else:
            logger.info("Fitting two HDBSCANs for ensemble")
            hdb_numerical_ = hdbscan.HDBSCAN(prediction_data=True, **self.hdbscan_params).fit(
                self.numerical_umap_.embedding_,
            )
            hdb_catergorical_ = hdbscan.HDBSCAN(prediction_data=True, **self.hdbscan_params).fit(
                self.categorical_umap_.embedding_,
            )
            self.hdbscan_ = {"hdb_numerical": hdb_numerical_, "hdb_categorical": hdb_catergorical_}

            # combine labels and probabilities for points
            predictions = self.combine_labels_and_probabilities(
                hdb_numerical_.labels_,
                hdb_numerical_.probabilities_,
                hdb_catergorical_.labels_,
                hdb_catergorical_.probabilities_,
            )

            self.labels_ = predictions[:, 0]
            self.probabilities_ = predictions[:, 1]

        logger.info("HDBSCAN fit")

    def score(self) -> np.array:
        """Returns the cluster assigned to each row.

        This is a wrapper function for HDBSCAN. It outputs the cluster labels
        that HDBSCAN converged on.

        Parameters
        ----------
        None : None

        Returns
        -------
        labels : np.array
        """
        return self.labels_

    def predict(self, df_new: pd.DataFrame) -> np.array:
        """
        Generate predictions on new data points for method 'ensemble'.
        Will use a weighted vote to pick the most representative cluster from two embeddings.

        Parameters
        ----------
        df_new : pd.DataFrame
            The new data for which to generate predictions.
            This should be a DataFrame with the same structure as the one used in the fit method.

        Returns
        -------
        labels : np.array
            The predicted cluster labels for each row in df_new.
         probabilities : np.array
            The  probabilities of the predictions for each row in df_new.
        """
        if self.umap_combine_method != "ensemble":
            raise ValueError('predict is only supported for method "ensemble"')

        numerical_values = extract_numerical(df_new, **self.kwargs)
        categorical_values = extract_categorical(df_new, **self.kwargs)

        # transform with umaps
        numerical_transform = self.numerical_umap_.transform(numerical_values)
        try:
            categorical_transform = self.categorical_umap_.transform(categorical_values)
        except ValueError as e:
            logger.error("Failed to transform categorical values: %s", str(e))
            raise ValueError(
                "Failed to transform categorical values",
                "This is most likely due to a lack of all categories in the data",
            ) from e  # pylint: disable=W0707

        # approximate predict
        numerical_labels, numerical_probabilities = hdbscan.approximate_predict(
            self.hdbscan_["hdb_numerical"],
            numerical_transform,
        )
        categorical_labels, categorical_probabilities = hdbscan.approximate_predict(
            self.hdbscan_["hdb_categorical"],
            categorical_transform,
        )

        # vote on cluster assignment
        predictions = self.combine_labels_and_probabilities(
            numerical_labels,
            numerical_probabilities,
            categorical_labels,
            categorical_probabilities,
        )

        return predictions

    @staticmethod
    def combine_labels_and_probabilities(
        numerical_labels: np.array,
        numerical_probabilities: np.array,
        categorical_labels: np.array,
        categorical_probabilities: np.array,
    ) -> np.array:
        """
         Combine labels and probabilities from two HDBSCAN models.

         Parameters
         ----------
         numerical_labels : np.ndarray
             Labels from the numerical HDBSCAN model.
         numerical_probabilities : np.ndarray
             Probabilities from the numerical HDBSCAN model.
         categorical_labels : np.ndarray
             Labels from the categorical HDBSCAN model.
         categorical_probabilities : np.ndarray
             Probabilities from the categorical HDBSCAN model.

         Returns
         -------
         labels : np.ndarray
             Combined labels.
         probabilities : np.ndarray
        Combined probabilities.
        """
        labels = np.where(
            numerical_labels == categorical_labels,
            numerical_labels,
            np.where(
                numerical_probabilities > categorical_probabilities,
                numerical_labels,
                categorical_labels,
            ),
        )
        probabilities = np.where(
            numerical_labels == categorical_labels,
            (numerical_probabilities + categorical_probabilities) / 2.0,  # normalize
            np.where(
                numerical_probabilities > categorical_probabilities,
                numerical_probabilities,
                categorical_probabilities,
            ),
        )
        return np.stack((labels, probabilities), axis=-1)
