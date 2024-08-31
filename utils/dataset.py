from typing import Dict, List

import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.data_utils import DataSample, get_avg_value
from utils.consts import CHEMICAL_SUBSTANCE_COLUMNS


class ChemicalDataset(Dataset):
    """
    A simple dataset.
    """

    def __init__(self, samples: List[DataSample]):
        samples = [sample for sample in samples if not math.isnan(sample.target_value)]
        self.samples = samples
        # We replace the nan value by this value.
        self.nan_default_replace_value = {substance: get_avg_value(substance=substance, samples=self.samples)
                                          for substance in CHEMICAL_SUBSTANCE_COLUMNS}
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.tensor(
            [sample.chem_substance_concentration[substance]
             if not math.isnan(sample.chem_substance_concentration[substance]) else
             self.nan_default_replace_value[substance] for substance in CHEMICAL_SUBSTANCE_COLUMNS],
            dtype=torch.float32)
        target = torch.tensor(sample.target_value, dtype=torch.float32)
        return features, target


def get_simple_dataloader(data_by_location: Dict[str, List[DataSample]], batch_size: int = 4,
                          shuffle: bool = False) -> DataLoader:
    """
    Get simple dataloader by combining data from all locations.
    :param data_by_location: Dictionary with the key being the location that the sample was taken and the value is list
    of samples taken from that location.
    :param batch_size: Batch size.
    :param shuffle: Whether to shuffle data.
    :return: Dataloader.
    """
    all_samples = [sample for location in sorted(list(data_by_location)) for sample in data_by_location[location]]
    dataset = ChemicalDataset(all_samples)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
