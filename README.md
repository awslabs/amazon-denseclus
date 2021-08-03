# Amazon DenseClus

DenseClus is a Python module for clustering mixed type data using [UMAP](https://github.com/lmcinnes/umap) and [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan). Allowing for both categorical and numerical data, DenseClus makes it possible to incoproate all features in clustering.

## Installation

```bash
pip install amazon-denseclus
```

## Usage

DenseClus requires a Panda's dataframe as input with both numerical and categorical columns.
All preprocessing and extraction are done under the hood, just call fit and then retrieve the clusters!

```python
from denseclus.DenseClus import DenseClus

clf = DenseClus(
    umap_combine_method="intersection_union_mapper",
)
clf.fit(df)

print(clf.score())
```

## Examples

A hands-on example with an overview of how to use is currently available in the form of a Jupyer notebook [here](https://github.com/smart-patrol/denseclus/blob/main/notebooks/DenseClus%20Example%20NB.ipynb).

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
