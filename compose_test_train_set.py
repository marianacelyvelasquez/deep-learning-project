import os
import shutil
import wfdb
import numpy as np
import pandas as pd
import skmultilearn
from pathlib import Path
from skmultilearn.model_selection import IterativeStratification
from utils.config import Config
from dataloaders.cinc2020.preprocess import Processor
from common.common import eq_classes
from dataloaders.cinc2020.common import labels_map


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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.join(script_dir, input_dir)

    record_paths: list[str] = []
    labels_binary_encoded_list: list[list[int]] = []

    for dirpath, dirnames, filenames in os.walk(absolute_path):
        filenames = [f for f in filenames if not f.startswith(".")]

        for filename in filenames:
            if filename.endswith(".hea"):
                record_path = os.path.join(dirpath, filename.split(".")[0])
                record_paths.append(record_path)

                # Get labels from the content of the .hea file
                try:
                    record = wfdb.rdrecord(record_path)
                    labels_binary_encoded = get_labels(record)
                    labels_binary_encoded_list.append(labels_binary_encoded)
                except ValueError as e:
                    print(f"Error reading record {record_path}: {e}")

    print(
        f"Record paths length: {len(record_paths)} \nLabels length: {len(labels_binary_encoded_list)} \n"
    )
    return record_paths, labels_binary_encoded_list


# data_source_dir_folder1: whole 2020 dataset, data_source_dir_folder2: 2021 chapman-shaoxing (CS) dataset
def compose_test_set(folder_A, folder_B):
    train_records = []
    test_records = []

    # Iterate over subfolders in folder A
    for subfolder_Ai in Path(folder_A).iterdir():
        if subfolder_Ai.is_dir():
            # Iterate over sub-subfolders in each A_i
            for subfolder in Path(subfolder_Ai).iterdir():
                if subfolder.is_dir():
                    record_paths, labels_binary_encoded = get_record_paths_and_labels_binary_encoded_list(subfolder)

                    X = np.array(record_paths)
                    y = np.array(labels_binary_encoded)

                    train_size = 0.95
                    test_size = 0.05
                    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[train_size, test_size])

                    for train_indexes, test_indexes in stratifier.split(X, y):
                        train_records.extend(X[train_indexes].tolist())
                        test_records.extend(X[test_indexes].tolist())

    # Add all records from folder B to the test set
    record_paths_folder_B, _ = get_record_paths_and_labels_binary_encoded_list(folder_B)
    test_records.extend(record_paths_folder_B)

# This is the part of the script that handles command line arguments
if __name__ == "__main__":
    import sys

    np.random.seed(42)

    if len(sys.argv) == 3:
        print("\n Entering pre-processing mode \n")
        data_source_dir_folder1 = sys.argv[1]
        data_source_dir_folder2 = sys.argv[2]

        train_data_dir = Config.TRAIN_DATA_DIR
        test_data_dir = Config.TEST_DATA_DIR

        if not os.path.exists(train_data_dir):
            os.makedirs(train_data_dir)

        if not os.path.exists(test_data_dir):
            os.makedirs(test_data_dir)

        # Call the new function to compose the stratified test set
        record_paths_train, record_paths_test = compose_test_set(data_source_dir_folder1, data_source_dir_folder2)

        # Move files to the specified directories
        for record_path in record_paths_train:
            shutil.move(record_path, train_data_dir)

        for record_path in record_paths_test:
            shutil.move(record_path, test_data_dir)
    else:
        print("Usage: python preprocess_dataset.py <data_source_dir_folder1> <data_source_dir_folder2>")
