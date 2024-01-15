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


def worker(device_id):
    while not task_queue.empty():
        try:
            fold_k, train_indices, validation_indices = task_queue.get_nowait()
        except queue.Empty:
            break

        print(
            f"Running fold {fold_k + 1}/{Config.NUM_FOLDS} on device {device_type}:{device_id}"
        )
        X_train = X[train_indices]
        y_train = y[train_indices]

        X_validation = X[validation_indices]
        y_validation = y[validation_indices]

        device = torch.device(f"{device_type}:{device_id}")

        experiment = dilatedCNNExperiment(
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
            classes,
            classes_test,
            fold_k,
            device,
            checkpoint_path,
        )

        if Config.ONLY_EVAL_TEST_SET:
            print("Only evaluating test set.")
            # TODO: We don't really need epoch for test evaluation no?
            # TODO: Actually we load validation and test and do the stratification and all.
            # In fact we don't need those.
            experiment.evaluate_test_set()
            break
        else:
            print("Running epochs.")
            experiment.run_epochs()  # Having CV_k go from 1 to #epochs


if __name__ == "__main__":
    device_type, device_count = get_device()

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

    stratifier = IterativeStratification(n_splits=Config.NUM_FOLDS, order=2)

    # Read optional checkpoint path
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Task Queue
    task_queue = queue.Queue()

    for fold_k, (train_indices, validation_indices) in enumerate(
        stratifier.split(X, y)
    ):
        task_queue.put((fold_k, train_indices, validation_indices))

        # If we only want to evaluate the test set, we only need to run one fold
        if Config.ONLY_EVAL_TEST_SET:
            break

    # Create and start threads
    threads = []
    for device_id in range(device_count):
        thread = threading.Thread(target=worker, args=(device_id,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # task_queue.join()
    print("Training complete!")
