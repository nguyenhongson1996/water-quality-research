from typing import List

import torch
from torch.utils.data import Dataset

from utils.data_utils import DataSample
from utils.consts import CHEMICAL_SUBSTANCE_COLUMNS


class ChemicalDataset(Dataset):
    """
    A simple dataset.
    """

    def __init__(self, samples: List[DataSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.tensor(
            [sample.chem_substance_concentration[substance] for substance in CHEMICAL_SUBSTANCE_COLUMNS],
            dtype=torch.float32)
        target = torch.tensor(sample.target_value, dtype=torch.float32)
        return features, target