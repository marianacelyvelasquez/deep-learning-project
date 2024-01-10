import warnings
import sys
import os
import torch
import numpy as np
import pandas as pd
from skmultilearn.model_selection import IterativeStratification

import utils.cinc_utils as cinc_utils
from preprocess_dataset import get_record_paths_and_labels_binary_encoded_list
from experiments.SWAG.SWAGInference import SWAGInference
from experiments.SWAG.SWAG import SWAGExperiment  # Import the new class
from experiments.SWAG.config import Config

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._ranking")

if __name__ == "__main__":
    print("Running SWAGExperiment.")

    # if os.path.isdir(Config.OUTPUT_DIR):
    #     print(
    #         "\033[91mOutput directory already exists. Either remove it or change it in the config file. Exiting.\033[0m"
    #     )
    #     quit()

    mapping = pd.read_csv("utils/label_mapping_ecgnet_eq.csv", delimiter=";")
    class_mapping = mapping[["SNOMED CT Code", "Training Code"]]
    X, y, classes = cinc_utils.get_xy(
        Config.TRAIN_DATA_DIR,
        max_sample_length=5000,
        cut_off=True,
        class_mapping=class_mapping,
    )

    X_test, y_test, classes_test = cinc_utils.get_xy(
        Config.TEST_DATA_DIR,
        max_sample_length=5000,
        cut_off=True,
        class_mapping=class_mapping,
    )

    classes = [str(c) for c in classes]
    classes_test = [str(c) for c in classes_test]

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None


    # TODO: Fix numpy seeds and torch seeds
    np.random.seed(42)
    torch.manual_seed(42)

    stratifier = IterativeStratification(n_splits=Config.NUM_FOLDS, order=2)

    for k, (train_indices, validation_indices) in enumerate(stratifier.split(X, y)):
        print(f"Running fold {k+1}/{Config.NUM_FOLDS}")
        X_train = X[train_indices]
        y_train = y[train_indices]

        X_validation = X[validation_indices]
        y_validation = y[validation_indices]


        # Create an instance of your SWAGExperiment class
        swag_experiment = SWAGExperiment(
            SWAGInference(
            checkpoint_path,
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
            classes,
            classes_test,
            CV_k=k,
            )
        )

        if Config.ONLY_EVAL_TEST_SET:
            print("Only evaluating test set.")
            # swag_experiment.evaluate_test_set() # TODO: Implement this
            break
        else:
            print("Running SWAG epochs.")
            swag_experiment.run()
