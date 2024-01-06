import os
import wfdb
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
from common.common import eq_classes
from dataloaders.cinc2020.common import labels_map

from skmultilearn.model_selection import IterativeStratification

"""
    This script splits data from the input dir into train and test sets.
    It does not process it. It creates symlinks.
"""


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
        # Filtering out hidden files
        filenames = [f for f in filenames if not f.startswith(".")]

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


# This is the part of the script that handles command line arguments
if __name__ == "__main__":
    np.random.seed(42)

    # Set up argparse for command line arguments
    parser = argparse.ArgumentParser(
        description="Preprocess dataset and split into train and test sets."
    )
    parser.add_argument(
        "data_source_dir", type=str, help="Directory of the source data"
    )
    parser.add_argument(
        "--tiny", action="store_true", help="Use a tiny dataset for quick testing"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/cinc2020_split",
        help="Output path for the split data",
    )

    # Parse arguments
    args = parser.parse_args()

    data_source_dir = Path(args.data_source_dir)
    data_source_dir = os.path.abspath(data_source_dir)

    train_data_dir = os.path.join(args.output_path, "train")
    train_data_dir = os.path.abspath(train_data_dir)

    test_data_dir = os.path.join(args.output_path, "test")
    test_data_dir = os.path.abspath(test_data_dir)

    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)

    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    print(f"Splitting data from {data_source_dir} into train and test sets...")
    print(f"Train data will be saved to {train_data_dir}")
    print(f"Test data will be saved to {test_data_dir}")

    print("\nGetting record paths and binary labels...")

    # Get lists of paths and labels
    (
        record_paths,
        labels_binary_encoded_list,
    ) = get_record_paths_and_labels_binary_encoded_list(data_source_dir)

    X = np.array(record_paths)
    y = np.array(labels_binary_encoded_list)

    if args.tiny is True:
        train_size = 0.1
        test_size = 0.05

        # Need to add up to 1.
        # We never use the 3th split.
        sample_distribution_per_fold = [
            train_size,
            test_size,
            1.0 - train_size - test_size,
        ]
        print("Tiny true.")
    else:
        train_size = 0.8
        test_size = 0.2

        sample_distribution_per_fold = [
            train_size,
            test_size,
        ]
        print("Tiny false.")

    stratifier = IterativeStratification(
        n_splits=len(sample_distribution_per_fold),
        order=2,
        sample_distribution_per_fold=sample_distribution_per_fold,
    )

    print(f"sample_distribution_per_fold: {sample_distribution_per_fold}")

    splits = []
    for _, set_indices in stratifier.split(X, y):
        # get X and y for this fold
        # print(f" set_indexes: {set_indexes} \n ")
        splits.append(set_indices)

    train_indices = splits[0]
    test_indices = splits[1]

    for idx in tqdm(train_indices, desc="Processing Training Data"):
        os.symlink(
            record_paths[idx] + ".hea",
            os.path.join(train_data_dir, Path(record_paths[idx]).stem + ".hea"),
        )

        os.symlink(
            record_paths[idx] + ".mat",
            os.path.join(train_data_dir, Path(record_paths[idx]).stem + ".mat"),
        )

    for idx in tqdm(test_indices, desc="Processing Testing Data"):
        os.symlink(
            record_paths[idx] + ".hea",
            os.path.join(test_data_dir, Path(record_paths[idx]).stem + ".hea"),
        )

        os.symlink(
            record_paths[idx] + ".mat",
            os.path.join(test_data_dir, Path(record_paths[idx]).stem + ".mat"),
        )
