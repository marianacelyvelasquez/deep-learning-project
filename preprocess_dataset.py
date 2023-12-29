import os
import shutil
import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
from skmultilearn.model_selection import IterativeStratification
from experiments.dilated_CNN.config import Config
from dataloaders.cinc2020.preprocess import Processor
from common.common import eq_classes

# MARI RIC: Assume _flattened exists :)



mappings = pd.read_csv("data/cinc2020/label_cinc2020_top24.csv", delimiter=",")
labels_map = mappings["SNOMED CT Code"].values


def make_directories_for_processed_split_data(source_dir: str) -> tuple[str, str, str]:
    processed_dir_training = Config.TRAIN_DATA_DIR
    processed_dir_training.mkdir(exist_ok=True)

    processed_dir_testing = Config.TEST_DATA_DIR
    processed_dir_testing.mkdir(exist_ok=True)

    #Validation set will no longer be used
    processed_dir_validating = Config.TEST_DATA_DIR
    processed_dir_validating.mkdir(exist_ok=True)

    return processed_dir_training, processed_dir_testing, processed_dir_validating


def get_labels(record) -> np.ndarray:
    diagnosis_string = record.comments[2].split(": ", 1)[1].strip()
    diagnosis_list = diagnosis_string.split(",")

    # Replace diagnosis with equivalent class if necessary
    for diagnosis in diagnosis_list:
        if diagnosis in eq_classes[:, 1]:
            eq_class = eq_classes[eq_classes[:, 1] == diagnosis][0][0]
            diagnosis_list = [eq_class if x == diagnosis else x for x in diagnosis_list]

    diagnosis_list = [int(diagnosis) for diagnosis in diagnosis_list]

    # Binary encode labels. 1 if label is present, 0 if not.
    return np.isin(labels_map, diagnosis_list).astype(int)


def get_record_paths_and_labels_binary_encoded_list(
    input_dir: str,
) -> tuple[list[str], list[list[int]]]:
    #
    # Get the directory containing the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = input_dir
    absolute_path = os.path.join(script_dir, relative_path)

    # Allocate memory for records list and labels list
    record_paths: list[str] = []
    labels_binary_encoded_list: list[list[int]] = []

    # Process each record and save to the appropriate directory
    for dirpath, dirnames, filenames in os.walk(absolute_path):
        for filename in filenames:
            if filename.endswith(".hea"):
                record_path = os.path.join(
                    dirpath, filename.split(".")[0]
                )  # keep path + filename without extension (so no difference between .hea and .mat)
                record_paths.append(record_path)
                # Get labels from the content of the .hea file [MARI: sorry I went fully functional]
                record = wfdb.rdrecord(record_path)
                labels_binary_encoded = get_labels(record)
                labels_binary_encoded_list.append(labels_binary_encoded)
    print(
        f"Record paths length: {len(record_paths)} \nLabels length: {len(labels_binary_encoded_list)} \n"
    )
    return record_paths, labels_binary_encoded_list


def iterative_train_test_split(X, y, train_size, n_splits=3):
    """Iteratively stratified train/test split

    Parameters
    ----------
    train_size : float, [0,1]
        the proportion of the dataset to include in the train split, the rest will be split between test and validation sets

    Returns
    -------
     X_train, y_train, X_test, y_test, X_val, y_val
        stratified division into train/test/validate split
    """
    test_size = (1.0 - train_size) / 2
    sample_distribution_per_fold = [train_size, test_size, 1 - train_size - test_size]

    stratifier = IterativeStratification(
        n_splits=n_splits,
        order=2,
        sample_distribution_per_fold=sample_distribution_per_fold,
    )

    splits = []
    for _, set_indexes in stratifier.split(X, y):
        # get X and y for this fold
        # print(f" set_indexes: {set_indexes} \n ")
        splits.append(set_indexes)

    train_indexes = splits[0]
    test_indexes = splits[1]
    val_indexes = splits[2]

    X_train = X[train_indexes]
    X_test = X[test_indexes]
    X_val = X[val_indexes]
    print(
        f"X_train shape: {X_train.shape} \n X_test shape: {X_test.shape} \n X_val shape: {X_val.shape} \n"
    )

    return X_train, X_test, X_val


def split_data_using_stratification(
    record_paths: list[str], labels_binary_encoded_list: list[list[int]]
) -> tuple[list[str], list[str], list[str]]:
    # Convert lists to arrays
    X = np.array(record_paths)
    y = np.array(labels_binary_encoded_list)
    # print(f"\n X shape: {X.shape}, y shape: {y.shape} \n")

    # Perform iterative stratified split
    X_train, X_test, X_val = iterative_train_test_split(
        X,
        y,
        train_size=0.5,  # 0.8 train ?
        n_splits=3,
    )

    # Convert arrays back to lists
    X_train = X_train.tolist()
    X_val = X_val.tolist()
    X_test = X_test.tolist()

    return X_train, X_test, X_val


def preprocess_dataset(source_dir: str):
    # First: create directories for the split data
    (
        processed_dir_training,
        processed_dir_testing,
        processed_dir_validating,
    ) = make_directories_for_processed_split_data(source_dir)

    # Get lists of paths and labels
    (
        record_paths,
        labels_binary_encoded_list,
    ) = get_record_paths_and_labels_binary_encoded_list(source_dir)

    # Now, stratifyyyyyyy :star:
    (
        record_paths_train,
        record_paths_test,
        record_paths_val,
    ) = split_data_using_stratification(record_paths, labels_binary_encoded_list)

    # print(
    #     f"\n\nRecord paths train length: {len(record_paths_train)} \nRecord paths test length: {len(record_paths_test)}, \nRecord paths val length: {len(record_paths_val)} \n"
    # )

    # Initialize the processor TODO check the paths are correct lol
    processor = Processor(source_dir)
    processor.process_records(record_paths_train, processed_dir_training)
    processor.process_records(record_paths_test, processed_dir_testing)
    processor.process_records(record_paths_val, processed_dir_validating)

    # # TODO: copy .mat files to directories
    # for filename in Path(input_dir).glob("*.mat"):
    #     dest_dir = None  # Determine the destination directory based on the filename or any other condition
    #     if dest_dir:
    #         copyfile(filename, Path(dest_dir) / filename.name)


# This is the part of the script that handles command line arguments
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        print("\n Entering pre-processing mode \n")
        data_source_dir = sys.argv[1]

        train_data_dir = Config.TRAIN_DATA_DIR
        test_data_dir = Config.TEST_DATA_DIR

        if not os.path.exists(train_data_dir):
            os.makedirs(train_data_dir)

        if not os.path.exists(test_data_dir):
            os.makedirs(test_data_dir)

        # Get lists of paths and labels
        (
            record_paths,
            labels_binary_encoded_list,
        ) = get_record_paths_and_labels_binary_encoded_list(data_source_dir)

        X = np.array(record_paths)
        y = np.array(labels_binary_encoded_list)

        train_size = 0.8
        test_size = 0.2
        sample_distribution_per_fold = [train_size, test_size]

        stratifier = IterativeStratification(
            n_splits=2,
            order=2,
            sample_distribution_per_fold=sample_distribution_per_fold,
        )

        splits = []
        for _, set_indexes in stratifier.split(X, y):
            # get X and y for this fold
            # print(f" set_indexes: {set_indexes} \n ")
            splits.append(set_indexes)

        train_indexes = splits[0]
        test_indexes = splits[1]

        X_train = X[train_indexes]
        X_test = X[test_indexes]

        processor = Processor()
        processor.process_records(X_train, train_data_dir)
        processor.process_records(X_test, test_data_dir)
    else:
        print("Usage: python preprocess_dataset.py <data_source_dir>")
