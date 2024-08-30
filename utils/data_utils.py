from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from utils.consts import DATA_DIR


def load_data_from_excels(files: List[str], data_dir: Path = DATA_DIR, ) -> pd.DataFrame:
    """
    Read data from excel files.
    :param files: Excel files that contain the data.
    :param data_dir: Folder that contains the data files.
    :return: Dataframe.
    """
    current_shape: Optional[Tuple[int, int]] = None
    current_columns: Optional[List[str]] = None
    dataframes: List[pd.DataFrame] = []
    for file in files:
        df = pd.read_excel(data_dir / file, header=0)
        if current_shape and df.shape != current_shape:
            raise ValueError(f"The shape of the dataframe is {df.shape} is missmatch with the current shape "
                             f"{current_shape}")
        else:
            current_shape = df.shape
        if current_columns and df.columns != current_columns:
            raise ValueError(f"The columns of the dataframe is {df.columns} is missmatch with the current column "
                             f"{current_columns}")
        else:
            current_columns = df.columns
        dataframes.append(df)
    return pd.concat(dataframes)
