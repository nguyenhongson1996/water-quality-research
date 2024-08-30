from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from utils.consts import COLUMNS, DATA_DIR


def load_data_from_excels(files: List[str], data_dir: Path = DATA_DIR, ) -> pd.DataFrame:
    """
    Read data from excel files.
    :param files: Excel files that contain the data.
    :param data_dir: Folder that contains the data files.
    :return: Dataframe.
    """
    current_columns: Optional[List[str]] = None
    dataframes: List[pd.DataFrame] = []
    for file in files:
        df = pd.read_excel(data_dir / file, header=0)
        assert set(COLUMNS).issubset(df.columns)
        df = df[COLUMNS]
        dataframes.append(df)
    return pd.concat(dataframes)
