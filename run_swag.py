import warnings
import sys
import os
import torch
import numpy as np
import pandas as pd
import importlib
from skmultilearn.model_selection import IterativeStratification

import utils.cinc_utils as cinc_utils
from preprocess_dataset import get_record_paths_and_labels_binary_encoded_list
from experiments.SWAG.SWAGInference import SWAGInference
from experiments.SWAG.SWAG import SWAGExperiment

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._ranking")

if __name__ == "__main__":
    print("Running SWAGExperiment.")

    config_file_name = sys.argv[1] if len(sys.argv) > 1 else 'config_lr_001'
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Import the corresponding Config file based on the learning rate
    config_module = importlib.import_module(f'experiments.SWAG.{config_file_name}')
    Config = config_module.Config

    k = 0  # No stratification


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



    # TODO: Fix numpy seeds and torch seeds
    np.random.seed(42)
    torch.manual_seed(42)

    stratifier = IterativeStratification(n_splits=Config.NUM_FOLDS, order=2)

    print(f"Running fold {k+1}/{Config.NUM_FOLDS}")

    # Create an instance of your SWAGExperiment class
    swag_experiment = SWAGExperiment(
        SWAGInference(
            checkpoint_path,
            X,
            y,
            X_test,
            y_test,
            classes,
            classes_test,
            Config,
            CV_k=k,
        )
    )

    if Config.ONLY_EVAL_TEST_SET:
        print("Only evaluating test set.")
        # swag_experiment.evaluate_test_set() # TODO: Implement this
    else:
        print("Running SWAG epochs.")
        swag_experiment.run()
