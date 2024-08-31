import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set the random seed for reproducibility.
    :param seed: The seed value to use. Default is 42.
    """
    # Set the random seed for Python's built-in random module.
    random.seed(seed)

    # Set the random seed for NumPy.
    np.random.seed(seed)

    # Set the random seed for PyTorch.
    torch.manual_seed(seed)

    # If CUDA (GPU) is used, set the random seed for it.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups.
        # These settings ensure deterministic behavior for CUDA.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set seed for any other libraries that use random operations.
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed()

PRJ_PATH = Path(os.path.abspath(__file__)).resolve().parent.parent

DATA_DIR = PRJ_PATH / "data"
WEIGHT_DIR = PRJ_PATH / "weights"

# Name of columns that is informative. We use these columns for training and evaluation.
COLUMNS = ["Name (E)", "YY/MM", "NH3-N(㎎/L)", "NO3-N(㎎/L)", "PO4-P(㎎/L)", "T-N(㎎/L)", "T-P(㎎/L)",
           "Dissolved Total N(㎎/L)", "Dissolved Total P(㎎/L)", "Hydrogen ion conc.", "DO (㎎/L)", "TSI(Chl-a)"]

# Some special columns.
LOCATION_COLUMN_NAME = "Name (E)"
TIME_COLUMN_NAME = "YY/MM"
TARGET_COLUMN_NAME = "TSI(Chl-a)"
# Columns that are the concentration of chemical substance.
CHEMICAL_SUBSTANCE_COLUMNS = list(set(COLUMNS) - {LOCATION_COLUMN_NAME, TIME_COLUMN_NAME})

# Year when the data was first collected.
START_YEAR = 2011
# Year when the data collection is finished.
END_YEAR = 2024
