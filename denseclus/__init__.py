"""
Dense Clus Library

Authors: Charles Frenzel, Baichaun Sun
Date: November 2023
"""

from .DenseClus import DenseClus
from .utils import extract_categorical, extract_numerical

__version__ = "0.1.2"

if __name__ == "__main__":  # pragma: no cover
    print(type(DenseClus), type(extract_categorical), type(extract_numerical))
