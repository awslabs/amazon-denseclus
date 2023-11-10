
# Amazon DenseClus

<p align="left">
<a href="https://github.com/awslabs/amazon-denseclus/actions/workflows/tests.yml"><img alt="build" src="https://github.com/awslabs/amazon-denseclus/actions/workflows/tests.yml/badge.svg"></a>
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
python3 -m pip install Amazon-DenseClus
```

## Quick Start

DenseClus requires a Panda's dataframe as input with both numerical and categorical columns.
All preprocessing and extraction are done under the hood, just call fit and then retrieve the clusters!

```python
from denseclus import DenseClus
from denseclus.utils import make_dataframe

df = make_dataframe()

clf = DenseClus()
clf.fit(df)

print(clf.score())
```

## Usage

For slower but more stable results select `intersection_union_mapper` to combine embedding layers via third UMAP.
Be sure that random seeds are set too!

```python
clf = DenseClus(
    umap_combine_method="intersection_union_mapper",
)
```

For advanced users, it's possible to select more fine-grained control of the underlying algorithms by passing
dictionaries into `DenseClus` class.

For example:
```python
from denseclus import DenseClus
from denseclus.utils import make_dataframe

umap_params = {'categorical': {'n_neighbors': 15, 'min_dist': 0.1},
              'numerical': {'n_neighbors': 20, 'min_dist': 0.1}}
hdbscan_params = {'min_cluster_size': 10}

df = make_dataframe()

clf = DenseClus(umap_combine_method="union"
              ,umap_params=umap_params
              ,hdbscan_params=hdbscan_params)

clf.fit(df)
```


## Examples

A hands-on example with an overview of how to use is currently available in the form of a [Jupyter Notebook](/notebooks/DenseClus%20Example%20NB.ipynb).

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
