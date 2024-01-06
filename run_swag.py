import warnings
import os
import torch
import numpy as np
from skmultilearn.model_selection import IterativeStratification

from preprocess_dataset import get_record_paths_and_labels_binary_encoded_list
from experiments.SWAG.SWAGInference import SWAGInference
from experiments.SWAG.SWAG import SWAGExperiment  # Import the new class
from experiments.SWAG.config import Config

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._ranking")

if __name__ == "__main__":
    print("Running SWAGExperiment.")

    if os.path.isdir(Config.OUTPUT_DIR):
        print(
            "\033[91mOutput directory already exists. Either remove it or change it in the config file. Exiting.\033[0m"
        )
        quit()

    # Load or preprocess your data as needed
    record_paths_train, labels_train = get_record_paths_and_labels_binary_encoded_list(
        Config.TRAIN_DATA_DIR
    )
    record_paths_test, labels_test = get_record_paths_and_labels_binary_encoded_list(
        Config.TEST_DATA_DIR
    )

    X = np.array(record_paths_train)
    y = np.array(labels_train)

    X_test = np.array(record_paths_test)
    y_test = np.array(labels_test)

    # TODO: Fix numpy seeds and torch seeds
    np.random.seed(42)
    torch.manual_seed(42)

    stratifier = IterativeStratification(n_splits=Config.NUM_FOLDS)

    for k, (train_indices, validation_indices) in enumerate(stratifier.split(X, y)):
        print(f"Running fold {k+1}/{Config.NUM_FOLDS}")
        X_train = X[train_indices]
        y_train = y[train_indices]

        X_validation = X[validation_indices]
        y_validation = y[validation_indices]


        # Create an instance of your SWAGExperiment class
        swag_experiment = SWAGExperiment(
            SWAGInference(
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,)
        )

        if Config.ONLY_EVAL_TEST_SET:
            print("Only evaluating test set.")
            # swag_experiment.evaluate_test_set() # TODO: Implement this
            break
        else:
            print("Running SWAG epochs.")
            swag_experiment.run()
