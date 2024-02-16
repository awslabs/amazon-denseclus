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
"""


import logging
import warnings
from importlib.util import find_spec
from typing import Union

import hdbscan
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.base import BaseEstimator, ClassifierMixin

from .categorical import extract_categorical
from .numerical import extract_numerical
from .utils import seed_everything


if find_spec("cuml"):
    from cuml.cluster import HDBSCAN as cuHDBSCAN  # pylint: disable=E0611, E0401
    from cuml.manifold.umap import UMAP as cuUMAP  # pylint: disable=E0611, E0401


logger = logging.getLogger("denseclus")
logger.setLevel(logging.ERROR)
sh = logging.StreamHandler()
sh.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)
logger.addHandler(sh)

# this suppresses the jaccard metric warning(s)
warnings.filterwarnings("ignore", category=UserWarning)

VALID_UMAP_COMBINE_METHODS = [
    "intersection",
    "union",
    "contrast",
    "intersection_union_mapper",
    "ensemble",
]


class DenseClus(BaseEstimator, ClassifierMixin):
    """DenseClus

    Creates UMAP embeddings and HDSCAN clusters from mixed data

    Parameters
    ----------
            random_state : int, default=42
                Random State for both UMAP and numpy.random.
                If set to None UMAP will run in Numba in multicore mode but
                results may vary between runs (when using CPU).
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

            use_gpu: bool, default=False
                If True UMAP and Hdbscan will use the rapidsai cuml library.
                This will lead to a significant speedup in the computation
                time during model creation. See: https://docs.rapids.ai/install
    """

    def __init__(
        self,
        random_state: int = 42,
        umap_combine_method: str = "intersection",
        verbose: bool = False,
        umap_params: dict = None,  # type: ignore # noqa
        hdbscan_params: dict = None,  # type: ignore # noqa
        use_gpu: bool = False,
        **kwargs,
    ):
        if use_gpu and umap_combine_method != "ensemble":
            raise ValueError("Only ensemble supported for GPU")

        self.random_state = random_state
        if isinstance(random_state, int):
            seed_everything(self.random_state)

        self.umap_combine_method = umap_combine_method

        # Default parameters
        default_umap_params = {
            "categorical": {
                # jaccard is an option but only takes sparse input
                "metric": "hamming",
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

        self._use_gpu = use_gpu

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

        self.kwargs = kwargs

    def __repr__(self):  # pragma: no cover
        return f"""DenseClus(random_state={self.random_state}
                            ,verbose={self.verbose}
                            ,umap_combine_method={self.umap_combine_method}
                            ,umap_params={self.umap_params}
                            ,hdbscan_params={self.hdbscan_params}
                            ,use_gpu = {self._use_gpu}
                            )"""

    def __str__(self):  # pragma: no cover
        return f"""DenseClus object (random_state={self.random_state}
                                    ,verbose={self.verbose}
                                    ,umap_combine_method={self.umap_combine_method}
                                    ,umap_params={self.umap_params}
                                    ,hdbscan_params={self.hdbscan_params}
                                    ,use_gpu = {self._use_gpu})"""

    @property
    def umap_combine_method(self):
        """Getter for umap combine"""
        return self._umap_combine_method

    @umap_combine_method.setter
    def umap_combine_method(self, value):
        """Setter for umap combine"""
        if value not in VALID_UMAP_COMBINE_METHODS:
            raise ValueError(f"Invalid umap_combine_method: {value}")
        self._umap_combine_method = value

    @property
    def random_state(self):
        """Getter for random state"""
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        """Setter for random state"""
        if value is not None and not isinstance(value, int):
            raise ValueError("random_state must be an integer or None")
        self._random_state = value

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
        self.__fit_categorical()

        logger.info("Fitting numerical UMAP")
        self.__fit_numerical()

        logger.info("Mapping/Combining Embeddings")
        if self.umap_combine_method != "ensemble":
            self._umap_embeddings()

        logger.info("Fitting HDBSCAN...")
        self._fit_hdbscan()

    @staticmethod
    def _fit_umap(data, parameters, use_gpu, random_state):
        """Fit a UMAP with the given parameters
        Args:
            data: the data to fit the UMAP on
            parameters: the parameters to use for the UMAP
            use_gpu: whether to use GPU or not
            random_state: random state to use for the UMAP. if null and using CPU will fit UMAP in parallel.
        Returns:
            the fitted UMAP
        """
        if random_state is None:
            logger.info("No random seed passed, running UMAP in Numba, parallel")

        if use_gpu:
            # pylint: disable=E0601
            logger.info("Fitting UMAP using GPU")
            return cuUMAP(
                verbose=False,
                **parameters,
            ).fit(data)

        logger.info("Fitting UMAP using CPU")
        return umap.UMAP(
            random_state=random_state,
            n_jobs=1 if random_state is not None else -1,
            verbose=False,
            low_memory=True,
            **parameters,
        ).fit(data)

    def __fit_numerical(self) -> None:
        """
        Fit a UMAP based on numerical data

        Returns:
            None
        """
        if self._use_gpu is False:
            logger.info("cuML not installed, using CPU")
        try:
            logger.info("Fitting UMAP for Numerical data")

            numerical_umap = self._fit_umap(
                data=self.numerical_,
                parameters=self.umap_params["numerical"],
                use_gpu=self._use_gpu,
                random_state=self.random_state,
            )

            self.numerical_umap_ = numerical_umap
            logger.info("Numerical UMAP fitted successfully")

        except Exception as e:
            logger.error("Failed to fit numerical UMAP: %s", str(e))
            raise

    def __fit_categorical(self) -> None:
        """
        Fit a UMAP based on categorical data

        Returns:
            None
        """
        if self._use_gpu is False:
            logger.info("cuML not installed, using CPU")

        try:
            logger.info("Fitting UMAP for Categorical data")

            categorical_umap = self._fit_umap(
                data=self.categorical_,
                parameters=self.umap_params["categorical"],
                use_gpu=self._use_gpu,
                random_state=self.random_state,
            )

            self.categorical_umap_ = categorical_umap
            logger.info("Categorical UMAP fitted successfully")

        except Exception as e:
            logger.error("Failed to fit categorical UMAP: %s", str(e))
            raise

    def _umap_embeddings(self) -> None:
        """Combines the numerical and categorical UMAP embeddings based on the specified method.

        Supported: 'intersection', 'union', 'contrast','intersection_union_mapper', and 'ensemble'

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

    @staticmethod
    def _fit_single_hdbscan(
        data: np.array,
        parameters: dict,
        prediction_data: bool = False,
        use_gpu: bool = False,
    ) -> hdbscan.HDBSCAN:
        """fit HDBSCAN to the provided embeddings

        Parameters
        ----------
            data (array-like): embeddings to cluster
            parameters (dict): hdbscan parameters
            prediction_data (bool): whether to generate extra cached data for predicting labels or membership vectors
            use_gpu (bool): whether to use GPU

        Returns
        -------
            cuml.cluster.HDBSCAN/hdbscan.HDBSCAN: fitted HDBSCAN
        """
        logger.info("Fitting HDBSCAN with parameters %s", parameters)

        if use_gpu:
            logger.info("Using GPU for HDBSCAN")
            # pylint: disable=E0601
            return cuHDBSCAN(
                prediction_data=prediction_data,
                **parameters,
            ).fit(data)

        logger.info("Using CPU for HDBSCAN")
        return hdbscan.HDBSCAN(
            prediction_data=prediction_data,
            **parameters,
        ).fit(data)

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

            hdb_ = self._fit_single_hdbscan(
                data=self.mapper_.embedding_,
                parameters=self.hdbscan_params,
                prediction_data=False,
                use_gpu=self._use_gpu,
            )

            self.hdbscan_ = hdb_
            self.labels_ = hdb_.labels_
            self.probabilities_ = hdb_.probabilities_

        else:
            logger.info("Fitting two HDBSCANs for ensemble")
            hdb_numerical_ = self._fit_single_hdbscan(
                data=self.numerical_umap_.embedding_,
                parameters=self.hdbscan_params,
                prediction_data=True,
                use_gpu=self._use_gpu,
            )
            hdb_catergorical_ = self._fit_single_hdbscan(
                data=self.categorical_umap_.embedding_,
                parameters=self.hdbscan_params,
                prediction_data=True,
                use_gpu=self._use_gpu,
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

    @staticmethod
    def combine_labels_and_probabilities(
        numerical_labels: Union[pd.Series, np.ndarray],
        numerical_probabilities: Union[pd.Series, np.ndarray],
        categorical_labels: Union[pd.Series, np.ndarray],
        categorical_probabilities: Union[pd.Series, np.ndarray],
    ) -> np.ndarray:
        """
        Combine labels and probabilities from two HDBSCAN models.

        Parameters
        ----------
        numerical_labels : Union[pd.Series, np.ndarray]
            Labels from the numerical HDBSCAN model.
        numerical_probabilities : Union[pd.Series, np.ndarray]
            Probabilities from the numerical HDBSCAN model.
        categorical_labels : Union[pd.Series, np.ndarray]
            Labels from the categorical HDBSCAN model.
        categorical_probabilities : Union[pd.Series, np.ndarray]
            Probabilities from the categorical HDBSCAN model.

        Returns
        -------
        labels : np.ndarray
            Combined labels
        probabilities : np.ndarray
            Combined probabilities
        """
        # explicit cast to np.array because labels can be series or np array
        numerical_labels = np.array(numerical_labels)
        numerical_probabilities = np.array(numerical_probabilities)
        categorical_labels = np.array(categorical_labels)
        categorical_probabilities = np.array(categorical_probabilities)

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

    def predict(self, df_new: pd.DataFrame) -> np.ndarray:
        """
        Predict the labels and probabilities for the new data.

        Parameters
        ----------
        df_new : pd.DataFrame
            The new data for which the labels and probabilities are to be predicted.

        Returns
        -------
        predictions : np.ndarray
            The predicted labels and probabilities for the new data.
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

        numerical_labels, numerical_probabilities = hdbscan.approximate_predict(
            self.hdbscan_["hdb_numerical"],
            numerical_transform,
        )
        categorical_labels, categorical_probabilities = hdbscan.approximate_predict(
            self.hdbscan_["hdb_categorical"],
            categorical_transform,
        )
        predictions = self.combine_labels_and_probabilities(
            numerical_labels,
            numerical_probabilities,
            categorical_labels,
            categorical_probabilities,
        )
        return predictions

    def evaluate(self, log_dbcv=False) -> np.array:
        """Evaluates the cluster and returns the cluster assigned to each row.

         This is a wrapper function for HDBSCAN. It outputs the cluster labels
         that HDBSCAN converged on.

         Parameters
         ----------
         log_dbcv (bool) : Whether to log DBCV scores. Defaults to False

        Returns
        -------
        labels : np.array
            The cluster labels assigned to each row.
        """
        labels = self.labels_
        clustered = labels >= 0

        if isinstance(self.hdbscan_, dict) or self.umap_combine_method == "ensemble":
            if log_dbcv:
                print(f"DBCV numerical score {self.hdbscan_['hdb_numerical'].relative_validity_}")
                print(
                    f"DBCV categorical score {self.hdbscan_['hdb_categorical'].relative_validity_}"
                )
            embedding_len = self.numerical_umap_.embedding_.shape[0]
            coverage = np.sum(clustered) / embedding_len
            print(f"Coverage {coverage}")
            return labels

        if log_dbcv:
            print(f"DBCV score {self.hdbscan_.relative_validity_}")
        embedding_len = self.mapper_.embedding_.shape[0]
        coverage = np.sum(clustered) / embedding_len
        print(f"Coverage {coverage}")
        return labels
