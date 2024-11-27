from typing import Dict, List
import math
import torch
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import DataSample, get_avg_value
from utils.consts import CHEMICAL_SUBSTANCE_COLUMNS

class ChemicalSequenceDataset(Dataset):
    """
    Dataset for LSTM: Returns sequences with a fixed length.
    """

    def __init__(self, samples: List[DataSample], seq_length: int):
        # Filter out all samples with NaN as target.
        samples = [sample for sample in samples if not math.isnan(sample.target_value)]
        self.samples = samples
        self.seq_length = seq_length  # Sequence length.

        # Replace NaN values in chemical concentrations with the corresponding average value.
        self.nan_default_replace_value = {
            substance: get_avg_value(substance=substance, samples=self.samples)
            for substance in CHEMICAL_SUBSTANCE_COLUMNS
        }
            # Ensure there are enough samples to create at least one full sequence.
        if len(self.samples) >= self.seq_length:
            # Create indices for non-overlapping sequences.
            self.indices = [0] + list(range(seq_length - 1, len(self.samples) - self.seq_length + 1, self.seq_length))
        else:
            # If not enough samples, set indices to an empty list.
            self.indices = []
    def __len__(self):
        # Total number of non-overlapping sequences available.
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the starting index for the current sequence.
        start_idx = self.indices[idx]
        
        if start_idx == 0:
            padding_sample = self.samples[0]  # Lấy giá trị tháng đầu tiên làm padding
            sequence_samples = [padding_sample] + self.samples[start_idx: start_idx + self.seq_length]
        else:
            # Extract a sequence of samples starting from `start_idx`.
            sequence_samples = self.samples[start_idx - 1: start_idx + self.seq_length]

        # Extract features for each sample in the sequence.
        features = torch.stack([
            torch.tensor(
                [
                    sample.chem_substance_concentration[substance] 
                    if not math.isnan(sample.chem_substance_concentration[substance])
                    else self.nan_default_replace_value[substance]
                    for substance in CHEMICAL_SUBSTANCE_COLUMNS
                ],
                dtype=torch.float32
            ) for sample in sequence_samples
        ])

        # Extract target values for the entire sequence.
        targets = torch.tensor(
            [sample.target_value for sample in sequence_samples], 
            dtype=torch.float32
        )  # Shape: (seq_length,).
        
        return features, targets

def get_lstm_dataloader(data_by_location: Dict[str, List[DataSample]], 
                        seq_length: int, batch_size: int = 4, 
                        shuffle: bool = False) -> DataLoader:
    """
    Create a DataLoader for LSTM.
    :param data_by_location: Dictionary with key being the location and value being the samples.
    :param batch_size: Batch size.
    :param seq_length: Sequence length.
    :param shuffle: Whether to shuffle the data.
    :return: Dataloader.
    """
    # Combine the samples from different locations.
    all_samples = [
        sample for location in sorted(list(data_by_location)) 
        for sample in data_by_location[location]
    ]
    
    # Create a dataset with a fixed sequence length.
    dataset = ChemicalSequenceDataset(all_samples, seq_length)

    # Create a DataLoader.
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader 