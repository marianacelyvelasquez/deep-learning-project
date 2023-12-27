import os
import shutil
import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
from skmultilearn.model_selection import iterative_train_test_split #TODO: understand why this is not working on my laptop
from dataloaders.cinc2020.preprocess import Processor


# MARI RIC: Assume _flattened exists :)
# input_dir = "data/2020_flattened"
input_dir = "data/2020_tiny"

eq_classes = np.array([
    ["713427006", "59118001"],
    ["284470004", "63593006"],
    ["427172004", "17338001"],
    ])

mappings = pd.read_csv("data/cinc2020/label_cinc2020_top24.csv", delimiter=",")
labels_map = mappings["SNOMED CT Code"].values


def make_directories_for_processed_split_data(source_dir: str) -> tuple[str, str, str]:
    processed_dir_training = Path(source_dir).parent / "cinc2020_processed_training"
    processed_dir_training.mkdir(exist_ok=True)

    processed_dir_testing = Path(source_dir).parent / "cinc2020_processed_testing"
    processed_dir_testing.mkdir(exist_ok=True)

    processed_dir_validating = Path(source_dir).parent / "cinc2020_processed_validating"
    processed_dir_validating.mkdir(exist_ok=True)

    return processed_dir_training, processed_dir_testing, processed_dir_validating

def get_record_paths_and_labels_binary_encoded_list(input_dir: str) -> tuple[list[str],list[list[int]]]:
    # Allocate mem for records list and labels list
    record_paths: list[str] = []
    labels_binary_encoded_list: list[list[int]] = []

    # Process each record and save to the appropriate directory
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(".hea"):
                record_path = os.path.join(dirpath, filename.split(".")[0]) # keep path + filename without extension (so no difference between .hea and .mat)
                record_paths.append(record_path)
                # Get labels from the content of the .hea file [MARI: sorry I went fully functional]
                record = wfdb.rdrecord(record_path)
                labels_binary_encoded = get_labels_binary_encoded(record)
                labels_binary_encoded_list.append(labels_binary_encoded)
    return record_paths, labels_binary_encoded_list

def split_data_using_stratification(record_paths: list[str], labels_binary_encoded_list: list[list[int]]) -> tuple[list[str],list[str],list[str]]:
    # Convert lists to arrays
    X = np.array(record_paths)
    y = np.array(labels_binary_encoded_list)

    # Perform iterative stratified split
    X_train, X_temp, y_train, y_temp = iterative_train_test_split(
        X,
        y,
        test_size=0.2,  # 0.8 train ?
        random_state=42,  # For reproducibility
        stratify=y
    )

    X_val, X_test, y_val, y_test = iterative_train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,  # Half test half validate
        random_state=42,  # For reproducibility
        stratify=y_temp
    )
    
    # Convert arrays back to lists
    X_train = X_train.tolist()
    X_val = X_val.tolist()
    X_test = X_test.tolist()

    return X_train, X_test, X_val

def get_labels_binary_encoded(record):
    diagnosis_string = record.comments[2].split(": ", 1)[1].strip()
    diagnosis_list = diagnosis_string.split(",")

    # Replace diagnosis with equivalent class if necessary
    for diagnosis in diagnosis_list:
        if diagnosis in eq_classes[:, 1]:
            eq_class = eq_classes[eq_classes[:, 1] == diagnosis][0][0]
            diagnosis_list = [
        eq_class if x == diagnosis else x for x in diagnosis_list
            ]

    diagnosis_list = [int(diagnosis) for diagnosis in diagnosis_list]

    # Binary encode labels. 1 if label is present, 0 if not.
    return np.isin(labels_map, diagnosis_list).astype(int)



def preprocess_dataset(source_dir: str):
    #First: create directories for the split data
    processed_dir_training, processed_dir_testing, processed_dir_validating = make_directories_for_processed_split_data(source_dir)

    # Get lists of paths and labels
    record_paths, labels_binary_encoded_list = get_record_paths_and_labels_binary_encoded_list(input_dir)

    # Now, stratifyyyyyyy :star:
    X_train, X_test, X_val = split_data_using_stratification(record_paths, labels_binary_encoded_list)
    
    # Initialize the processor TODO check the paths are correct lol
    processor = Processor(input_dir)
    processor.process_records(X_train, processed_dir_training)
    processor.process_records(X_test, processed_dir_testing)
    processor.process_records(X_val, processed_dir_validating)


    # # TODO: copy .mat files to directories
    # for filename in Path(input_dir).glob("*.mat"):
    #     dest_dir = None  # Determine the destination directory based on the filename or any other condition
    #     if dest_dir:
    #         copyfile(filename, Path(dest_dir) / filename.name)



# This is the part of the script that handles command line arguments
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        print("Entering pre-processing mode")
        preprocess_dataset(sys.argv[1])
    else:
        print("Usage: python preprocess_dataset.py <path_to_cinc2020_directory>")


    
