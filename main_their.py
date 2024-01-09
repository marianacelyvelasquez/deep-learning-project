import warnings
import sys
import os
import numpy as np
from skmultilearn.model_selection import IterativeStratification
import pandas as pd
import utils.cinc_utils as cinc_utils

from preprocess_dataset import get_record_paths_and_labels_binary_encoded_list
from experiments.dilated_CNN.dilatedCNN import dilatedCNNExperiment
from experiments.dilated_CNN.config import Config

# We get a lot of warnings from sklearn.metrics._ranking (computing average precision)
# because we have lots of samples without any positive labels because the
# data has 111 classes but we only look at 24.
# We can ignore these warnings.
warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.metrics._ranking"
)

if __name__ == "__main__":
    print("Running dialted_CNN experiment.")

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

    for k, (train_indices, validation_indices) in enumerate(stratifier.split(X, y)):
        print(f"Running fold {k+1}/10")
        X_train = X[train_indices]
        y_train = y[train_indices]

        X_validation = X[validation_indices]
        y_validation = y[validation_indices]

        experiment = dilatedCNNExperiment(
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
            classes,
            classes_test,
            k + 1,
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
