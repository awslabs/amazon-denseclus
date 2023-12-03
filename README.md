
# Amazon DenseClus

<p align="left">
<a href="https://github.com/awslabs/amazon-denseclus/actions/workflows/tests.yml"><img alt="build" src="https://github.com/awslabs/amazon-denseclus/actions/workflows/cd.yml/badge.svg"></a>
<a><img alt="total download" src="https://static.pepy.tech/personalized-badge/amazon-denseclus?period=total&units=international_system&left_color=black&right_color=green&left_text=Total Downloads"></a>
<a><img alt="month download" src="https://static.pepy.tech/personalized-badge/amazon-denseclus?period=month&units=international_system&left_color=black&right_color=green&left_text=Monthly Downloads"></a>
<a><img alt="weekly download" src="https://static.pepy.tech/personalized-badge/amazon-denseclus?period=week&units=international_system&left_color=black&right_color=green&left_text=Weekly Downloads"></a>
<a href="https://badge.fury.io/py/Amazon-DenseClus"><img alt="PyPI version" src="https://badge.fury.io/py/Amazon-DenseClus.svg"></a>
<a><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/Amazon-DenseClus"></a>
<a><img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/Amazon-DenseClus"></a>
<a><img alt="PyPI - License" src="https://img.shields.io/pypi/l/Amazon-DenseClus"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/marketplace/actions/super-linter"><img alt="Github Super-Linter" src="https://github.com/awslabs/amazon-denseclus/workflows/Lint%20Code%20Base/badge.svg"></a>
</p>



DenseClus is a Python module for clustering mixed type data using [UMAP](https://github.com/lmcinnes/umap) and [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan). Allowing for both categorical and numerical data, DenseClus makes it possible to incorporate all features in clustering.

## Installation

```bash
python3 -m pip install amazon-denseclus
```

## Quick Start

DenseClus requires a Panda's dataframe as input with both numerical and categorical columns.
All preprocessing and extraction are done under the hood, just call fit and then retrieve the clusters!

```python
from denseclus import DenseClus
from denseclus.utils import make_dataframe


df = make_dataframe()
clf = DenseClus(df)
clf.fit(df)

scores = clf.score()
print(scores[0:10])
```


## Usage

### Prediction

DenseClus uses a `predict` method whhne `umap_combine_method` is set to `ensemble`.
Results are return in 2d array with the first part being the labels and the second part the probabilities.

```python
from denseclus import DenseClus
from denseclus.utils import make_dataframe

RANDOM_STATE = 10

df = make_dataframe(random_state=RANDOM_STATE)
train = df.sample(frac=0.8, random_state=RANDOM_STATE)
test = df.drop(train.index)
clf = DenseClus(random_state=RANDOM_STATE, umap_combine_method='ensemble')
clf.fit(train)

predictions = clf.predict(test)
print(predictions) # labels, probabilities
```


### On Combination Method

For a slower but more **stable** results select `intersection_union_mapper` to combine embedding layers via a third UMAP, which will provide equal weight to both numerics and categoriel columns. By default, you are setting the random seed which eliminates the ability for UMAP to run in parallel but will help circumevent some of [the randomness](https://umap-learn.readthedocs.io/en/latest/reproducibility.html) of the algorithm.

```python
clf = DenseClus(
    umap_combine_method="intersection_union_mapper",
)
```


### Advanced Usage

For advanced users, it's possible to select more fine-grained control of the underlying algorithms by passing
dictionaries into `DenseClus` class for either UMAP or HDBSCAN.

For example:
```python
from denseclus import DenseClus
from denseclus.utils import make_dataframe

umap_params = {
    "categorical": {"n_neighbors": 15, "min_dist": 0.1},
    "numerical": {"n_neighbors": 20, "min_dist": 0.1},
}
hdbscan_params = {"min_cluster_size": 10}

df = make_dataframe()

clf = DenseClus(umap_combine_method="union"
             , umap_params=umap_params
             , hdbscan_params=hdbscan_params
             , random_state=None) # this will run in parallel

clf.fit(df)
```


## Examples

### Notebooks

A hands-on example with an overview of how to use is currently available in the form of a [Example Jupyter Notebook](/notebooks/01_DenseClusExampleNB.ipynb).

Should you need to tune HDBSCAN, here is an optional approach: [Tuning with HDBSCAN Notebook](/notebooks/02_TuningwithHDBSCAN.ipynb)

Should you need to validate UMAP emeddings, there is an approach to do so in the [Validation for UMAP Notebook](/notebooks/03_ValidationForUMAP.ipynb)

### Blogs


[AWS Blog: Introducing DenseClus, an open source clustering package for mixed-type data](https://aws.amazon.com/blogs/opensource/introducing-denseclus-an-open-source-clustering-package-for-mixed-type-data/)

[TDS Blog: How To Tune HDBSCAN](https://towardsdatascience.com/tuning-with-hdbscan-149865ac2970)

[TDS Blog: On the Validation of UMAP](https://towardsdatascience.com/on-the-validating-umap-embeddings-2c8907588175)



## References

```bibtex
@article{mcinnes2018umap-software,
  title={UMAP: Uniform Manifold Approximation and Projection},
  author={McInnes, Leland and Healy, John and Saul, Nathaniel and Grossberger, Lukas},
  journal={The Journal of Open Source Software},
  volume={3},
  number={29},
  pages={861},
  year={2018}
}
```

```bibtex
@article{mcinnes2017hdbscan,
  title={hdbscan: Hierarchical density based clustering},
  author={McInnes, Leland and Healy, John and Astels, Steve},
  journal={The Journal of Open Source Software},
  volume={2},
  number={11},
  pages={205},
  year={2017}
}
```
