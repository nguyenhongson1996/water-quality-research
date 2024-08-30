import os
from pathlib import Path

PRJ_PATH = Path(os.path.abspath(__file__)).resolve().parent.parent

DATA_DIR = PRJ_PATH / "data"
WEIGHT_DIR = PRJ_PATH / "weights"

# Name of columns that is informative. We use these columns for training and evaluation.
COLUMNS = ['Name (E)', 'YY/MM', 'NH3-N(㎎/L)', 'NO3-N(㎎/L)', 'PO4-P(㎎/L)', 'T-N(㎎/L)', 'T-P(㎎/L)',
           'Dissolved Total N(㎎/L)', 'Dissolved Total P(㎎/L)', 'Hydrogen ion conc.', 'DO (㎎/L)', 'TSI(Chl-a)']
