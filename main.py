import warnings
import numpy as np
from skmultilearn.model_selection import IterativeStratification

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

    stratifier = IterativeStratification(n_splits=10)

    for k, (train_indices, validation_indices) in enumerate(stratifier.split(X, y)):
        print(f"Running fold {k+1}/10")
        X_train = X[train_indices]
        X_validation = X[validation_indices]

        y_train = y[train_indices]
        y_validation = y[validation_indices]

        experiment = dilatedCNNExperiment(
            X_train, y_train, X_validation, y_validation, X_test, y_test
        )
        experiment.run_epochs()
