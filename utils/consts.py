import os
from pathlib import Path

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
