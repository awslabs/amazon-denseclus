"""
Dense Clus Library
"""

from .categorical import extract_categorical
from .DenseClus import DenseClus
from .numerical import extract_numerical

if __name__ == "__main__":  # pragma: no cover
    print(type(DenseClus), type(extract_categorical), type(extract_numerical))
