import sys
import os
import numpy as np
from skmultilearn.model_selection import IterativeStratification
import pandas as pd
import utils.cinc_utils as cinc_utils
from utils.get_device import get_device
import queue
import threading
import torch

from preprocess_dataset import get_record_paths_and_labels_binary_encoded_list
from experiments.dilated_CNN.dilatedCNN import dilatedCNNExperiment
from experiments.dilated_CNN.config import Config

##
# THIS IS THE MAIN FILE FOR THE DILATED CNN EXPERIMENT FROM THE PAPER CODE
##

if __name__ == "__main__":
    device_type, device_count = get_device()
    device = torch.device(f"{device_type}:0")

    print("Running dialted_CNN experiment.")
    print("")
    print(f" > Found {device_count} {device_type} devices.")

    if os.path.isdir(Config.OUTPUT_DIR):
        print(
            "\033[91mOutput directory already exists. Either remove it or change it in the config file. Exiting.\033[0m"
        )
        quit()

    mapping = pd.read_csv("utils/label_mapping_ecgnet_eq.csv", delimiter=";")
    class_mapping = mapping[["SNOMED CT Code", "Training Code"]]
    X_train, y_train, classes = cinc_utils.get_xy(
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

    # Read optional checkpoint path
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None

    experiment = dilatedCNNExperiment(
        X_train,
        y_train,
        X_test,  # this is acutally validation but we just pass test because we dont care
        y_test,  # this is acutally validation but we just pass test because we dont care
        X_test,
        y_test,
        classes,
        classes_test,
        0,
        device,
        checkpoint_path,
    )

    if Config.ONLY_EVAL_TEST_SET:
        print("Only evaluating test set.")
        # TODO: We don't really need epoch for test evaluation no?
        # TODO: Actually we load validation and test and do the stratification and all.
        # In fact we don't need those.
        experiment.evaluate_test_set()
    else:
        print("Running epochs.")
        experiment.run_epochs()  # Having CV_k go from 1 to #epochs

    print("Training complete!")
