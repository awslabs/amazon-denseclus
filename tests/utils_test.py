import pandas as pd
import os

from denseclus.utils import seed_everything, make_dataframe


def test_seed_everything():
    seed_everything(42)
    assert os.getenv("PYTHONHASHSEED") == "42"


def test_make_dataframe():
    df = make_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, 10)
    assert df.columns[0] == "num_0"
    assert df.columns[9] == "cat_3"
    assert df.index[0] == 0
    assert df.index[9] == 9
