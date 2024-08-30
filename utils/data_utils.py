from bisect import bisect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.consts import (CHEMICAL_SUBSTANCE_COLUMNS, COLUMNS, DATA_DIR, END_YEAR, LOCATION_COLUMN_NAME, START_YEAR,
                          TARGET_COLUMN_NAME, TIME_COLUMN_NAME)


class DataSample(NamedTuple):
    """
    An instance store a sample of data.
    'location' is the location that the sample was collected.
    'year' and 'month' is the time that the sample was collected.
    'chem_substance_concentration' is a dictionary with the key being the chemical substance and the value being the
    corresponding concentration.
    'target_value' is the concentration of 'TSI(Chl-a)'.
    """
    location: str
    year: int
    month: int
    chem_substance_concentration: Dict[str, Any]
    target_value: float


def get_month_year(time_str: str) -> Optional[Tuple[int, int]]:
    """
    Split the data collection time into month and year.
    :param time_str: The time in string format when the data was collected.
    :return: Year and Month when the data was collected.
    """
    if not isinstance(time_str, str):
        return None
    time_splited = time_str.split("/")
    if len(time_splited) != 2 or int(time_splited[0]) < START_YEAR or int(time_splited[0]) > END_YEAR:
        # The time is invalid.
        return None
    year = int(time_splited[0])
    month = int(time_splited[1])
    if month < 1 or month > 12:
        return None
    return year, month


def load_data_from_excels(files: List[str], data_dir: Path = DATA_DIR, ) -> List[DataSample]:
    """
    Read data from excel files.
    :param files: Excel files that contain the data.
    :param data_dir: Folder that contains the data files.
    :return: Dataframe.
    """
    dataframes: List[pd.DataFrame] = []
    for file in files:
        df = pd.read_excel(data_dir / file, header=0)
        assert set(COLUMNS).issubset(df.columns)
        df = df[COLUMNS]
        dataframes.append(df)
    full_df = pd.concat(dataframes)

    full_data: List[DataSample] = []
    for index, row in tqdm(full_df.iterrows()):
        location = row[LOCATION_COLUMN_NAME]
        time = row[TIME_COLUMN_NAME]
        data_collection_time = get_month_year(time)
        if not data_collection_time:
            print(f"The time {time} in the row {index}th is invalid.")
            continue
        target = row[TARGET_COLUMN_NAME]
        if target == np.nan:
            print(f"The target value in the row {index}th is invalid.")
            continue

        chem_substance_concentration = {chemical_substance: row[chemical_substance] for chemical_substance in
                                        CHEMICAL_SUBSTANCE_COLUMNS}

        full_data.append(DataSample(location=location, year=data_collection_time[0], month=data_collection_time[1],
                                    chem_substance_concentration=chem_substance_concentration, target_value=target))
    return full_data


def split_data_by_location(full_data: List[DataSample]) -> Dict[str, List[DataSample]]:
    """
    Split data into smaller datasets by the location. For each sub-dataset, we sort the samples by the collection time.
    :param full_data: All data.
    :return: Sub-datasets each corresponds to a location.
    """
    data_by_location: Dict[str, List[DataSample]] = {}
    for sample in tqdm(full_data):
        data_by_location.setdefault(sample.location, []).append(sample)
    for location in data_by_location:
        data_by_location[location] = sorted(data_by_location[location], key=lambda sample: (sample.year, sample.month))
        start_sample = data_by_location[location][0]
        end_sample = data_by_location[location][-1]
        print(f"There are {len(data_by_location[location])} samples collected at '{location}' from "
              f"{start_sample.year}/{start_sample.month} to {end_sample.year}/{end_sample.month}")
    return data_by_location


def split_train_test(full_data: Dict[str, List[DataSample]],
                     start_test_year: int) -> Tuple[Dict[str, List[DataSample]], Dict[str, List[DataSample]]]:
    """
    Split the full data into training and testing dataset. Due to the nature of the data, the testing set is selected
    from the tail.
    :param full_data: Full data.
    :param start_test_year: The starting year of testing set. All samples from this year are considered as testing set
    and all samples from the previous years are considered as training set.
    :return: Training and testing dataset.
    """
    min_accepted_test_year = max(
        min(sample.year for sample in data_by_location[location]) for location in data_by_location)
    if start_test_year < min_accepted_test_year + 1:
        raise ValueError(f"The testing set must be selected after {start_test_year}")
    training_data: Dict[str, List[DataSample]] = {}
    testing_data: Dict[str, List[DataSample]] = {}
    for location in full_data:
        start_test_index = bisect([sample.year for sample in full_data[location]], start_test_year)
        print(f"Split data at {location} into {start_test_index} training samples and "
              f"{len(full_data[location]) - start_test_index} testing samples")
        training_data[location] = full_data[location][:start_test_index]
        testing_data[location] = full_data[location][start_test_index:]
    return training_data, testing_data


if __name__ == "__main__":
    files = ["predata.xls", "Data.xlsx"]
    data = load_data_from_excels(files)
    print(len(data))
    data_by_location = split_data_by_location(data)
    training_data, testing_data = split_train_test(data_by_location, start_test_year=2018)
