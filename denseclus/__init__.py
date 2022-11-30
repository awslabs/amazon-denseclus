from .DenseClus import DenseClus
from .utils import extract_categorical, extract_numerical

__version__ = "0.0.19"

if __name__ == "__main__":
    print(type(DenseClus), type(extract_categorical), type(extract_numerical))
