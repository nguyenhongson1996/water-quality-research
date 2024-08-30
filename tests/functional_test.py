import pytest
from typing import Callable, Dict, Any, List
from utils.data_utils import load_data_from_excels


class TestRules:
    def test_load_data(self):
        files = ["predata.xls", "Data.xlsx"]
        df = load_data_from_excels(files)
        print(df.shape)
        pass