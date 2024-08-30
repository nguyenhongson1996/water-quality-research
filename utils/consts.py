import os
from pathlib import Path

PRJ_PATH = Path(os.path.abspath(__file__)).resolve().parent.parent

DATA_DIR = PRJ_PATH / "data"
WEIGHT_DIR = PRJ_PATH / "weights"

