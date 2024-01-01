import csv
import os
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from sklearn.metrics import multilabel_confusion_matrix, average_precision_score
from experiments.dilated_CNN.config import Config

from utils.evaluate_12ECG_score import load_weights, compute_challenge_metric

from dataloaders.cinc2020.common import labels_map
from common.common import eq_classes


class MetricsManager:
    def __init__(self, name, num_epochs, num_classes, num_batches, CV_k):
        # True Positives
        self.tp: npt.NDArray[np.int_] = np.zeros((num_epochs, num_classes))
        # False Negatives
        self.fn: npt.NDArray[np.int_] = np.zeros((num_epochs, num_classes))
        # True Positives(
        self.fp: npt.NDArray[np.int_] = np.zeros((num_epochs, num_classes))
        # True Negatives
        self.tn: npt.NDArray[np.int_] = np.zeros((num_epochs, num_classes))

        self.loss: npt.NDArray[np.float_] = np.zeros((num_epochs, num_batches))

        self.output_folder = os.path.join(Config.OUTPUT_DIR, f"fold_{CV_k}", "metrics")
        self.output_filename = f"{name}_metrics.csv"

        if not os.path.exists(os.path.dirname(self.output_folder)):
            os.makedirs(os.path.dirname(self.output_folder))

        # TODO: Implement
        # Challenge score
        self.challenge_metric: npt.NDArray[np.float_] = np.zeros(num_epochs)

        # For each class, we have a list of metrics for each epoch.
        # You can take the average over all classes for a given epoch
        # to get the "macro" average.

        # Recall
        self.recall: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        # True Negative Rate
        self.TNR: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        # False Positive Rate
        self.FPR: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        # Positive Predictive Value
        self.PPV: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        # Negative Predictive Value
        self.NPV: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        # F1 score
        self.f1: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        # F2 score
        self.f2: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        # G2 score
        self.g2: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        # Receiver Operating Characteristic Area Under the Curve
        self.ROCAUC: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        # Average Precision
        self.AP: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))

        # Micro averages
        self.recall_micro: npt.NDArray[np.float_] = np.zeros(num_epochs)
        self.TNR_micro: npt.NDArray[np.float_] = np.zeros(num_epochs)
        self.FPR_micro: npt.NDArray[np.float_] = np.zeros(num_epochs)
        self.PPV_micro: npt.NDArray[np.float_] = np.zeros(num_epochs)
        self.NPV_micro: npt.NDArray[np.float_] = np.zeros(num_epochs)
        self.f1_micro: npt.NDArray[np.float_] = np.zeros(num_epochs)
        self.f2_micro: npt.NDArray[np.float_] = np.zeros(num_epochs)
        self.g2_micro: npt.NDArray[np.float_] = np.zeros(num_epochs)
        self.ROCAUC_micro: npt.NDArray[np.float_] = np.zeros(num_epochs)
        self.AP_micro: npt.NDArray[np.float_] = np.zeros(num_epochs)
        self.loss_micro: npt.NDArray[np.float_] = np.zeros(num_epochs)  # TODO WHYYYY

    def update_confusion_matrix(self, y_trues, y_preds, epoch):
        """
        Update the metric counts based on a batch of predictions, labels and loss.

        Args:
        :param predictions: Predicted outputs of the model for the batch. Dimension: (batch_size, num_classes)
        :param labels: Actual labels for the batch. Dimension: (batch_size, num_classes)
        :param loss: Loss value for the batch.
        """

        # Move data to RAM so we can compute stuff with it.
        y_trues = y_trues.detach().cpu()
        y_preds = y_preds.detach().cpu()

        # Format: https://scikit-learn.org/stable/glossary.html#term-multilabel-indicator-matrices
        # Gives us a list of confusion matrices for each class and for the whole batch
        confusion_matrices = multilabel_confusion_matrix(y_trues, y_preds)

        # TODO: Check computation of tp, fn etc. manually again and make sure it's correct 100%
        for class_i, confusion_matrix in enumerate(confusion_matrices):
            tn, fp, fn, tp = confusion_matrix.ravel()

            self.tp[epoch][class_i] += tp
            self.fn[epoch][class_i] += fn
            self.fp[epoch][class_i] += fp
            self.tn[epoch][class_i] += tn

        # Compute AP here because we don't have access to y_true and y_pred after
        self.AP[epoch] = average_precision_score(y_trues, y_preds)
        self.AP_micro[epoch] = average_precision_score(
            y_trues, y_preds, average="micro"
        )

    def update_loss(self, loss, epoch, batch_i):
        self.loss[epoch][batch_i] = loss.detach().cpu()

    def compute_micro_averages(self, epoch):
        tp = np.sum(self.tp[epoch])
        fn = np.sum(self.fn[epoch])
        fp = np.sum(self.fp[epoch])
        tn = np.sum(self.tn[epoch])

        self.recall_micro[epoch] = self.compute_recall(tp, fn)
        self.TNR_micro[epoch] = self.compute_TNR(tn, fp)
        self.FPR_micro[epoch] = self.compute_false_positive_rate(tn, fp)
        self.PPV_micro[epoch] = self.compute_positive_predictive_value(tp, fp)
        self.NPV_micro[epoch] = self.compute_negative_predictive_value(tn, fn)
        self.f1_micro[epoch] = self.compute_f1(
            self.recall_micro[epoch], self.PPV_micro[epoch]
        )
        self.f2_micro[epoch] = self.compute_fbeta(tp, fp, fn, beta=2)
        self.g2_micro[epoch] = self.compute_jaccard(tp, fp, fn, beta=2)
        self.loss_micro[epoch] = np.mean(self.loss[epoch])

    #  TODO: We work with 24 clases,A has 27. or i think 27 are used in the paper code???
    def compute_challenge_metric(self, y_true, y_pred, epoch):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        classes = [str(class_) for class_ in labels_map]
        weights_file = "utils/reward_matrix_weights.csv"
        weights = load_weights(weights_file, classes)

        normal_class = "426783006"

        challenge_score = compute_challenge_metric(
            weights, y_true, y_pred, list(classes), normal_class
        )

        self.challenge_metric[epoch] = challenge_score

    def compute_metrics(self, epoch):
        current_tp = self.tp[epoch]
        current_tn = self.tn[epoch]
        current_fp = self.fp[epoch]
        current_fn = self.fn[epoch]

        self.recall[epoch] = self.compute_recall(current_tp, current_fn)
        self.TNR[epoch] = self.compute_TNR(current_tn, current_fp)
        self.FPR[epoch] = self.compute_false_positive_rate(current_tn, current_fp)
        self.PPV[epoch] = self.compute_positive_predictive_value(current_tp, current_fp)
        self.NPV[epoch] = self.compute_negative_predictive_value(current_tn, current_fn)
        self.f1[epoch] = self.compute_f1(self.recall[epoch], self.PPV[epoch])
        self.f2[epoch] = self.compute_fbeta(current_tp, current_fp, current_fn, beta=2)
        self.g2[epoch] = self.compute_jaccard(
            current_tp, current_fp, current_fn, beta=2
        )
        # TODO: Implement average precision score

        print("Updated all metrics for the current epoch.")

    @staticmethod
    def compute_recall(tp, fn):
        # Calculate recall for each class
        recall = tp / (tp + fn)
        recall = np.nan_to_num(recall)

        return recall

    @staticmethod
    def compute_TNR(tn, fp):
        TNR = tn / (tn + fp)
        TNR = np.nan_to_num(TNR)

        return TNR

    @staticmethod
    def compute_false_positive_rate(tn, fp):
        false_positive_rate = fp / (tn + fp)
        false_positive_rate = np.nan_to_num(false_positive_rate)

        return false_positive_rate

    @staticmethod
    def compute_positive_predictive_value(tp, fp):
        positive_predictive_value = tp / (tp + fp)
        positive_predictive_value = np.nan_to_num(positive_predictive_value)

        return positive_predictive_value

    @staticmethod
    def compute_negative_predictive_value(tn, fn):
        negative_predictive_value = tn / (tn + fn)
        negative_predictive_value = np.nan_to_num(negative_predictive_value)

        return negative_predictive_value

    @staticmethod
    def compute_f1(recall, precision):
        f1 = 2 * ((recall * precision) / (recall + precision))
        f1 = np.nan_to_num(f1)

        return f1

    @staticmethod
    def compute_fbeta(tp, fp, fn, beta=2):
        fbeta = ((1 + beta**2) * tp) / ((1 + beta**2) * tp + fp + (beta**2) * fn)
        fbeta = np.nan_to_num(fbeta)

        return fbeta

    @staticmethod
    def compute_jaccard(tp, fp, fn, beta=2):
        jaccard = tp / (tp + fp + beta * fn)
        jaccard = np.nan_to_num(jaccard)

        return jaccard

    def report(self, epoch):
        # Define the metrics to report
        metrics = [
            "recall",
            "TNR",
            "FPR",
            "PPV",
            "NPV",
            "F1",
            "F2",
            "Jaccard",
            "AP",
            "loss",
        ]
        num_classes = 24  # Assuming 24 classes

        # Print the header
        header = f"{'Class':>6} | " + " | ".join(
            [f"{metric:>10}" for metric in metrics]
        )
        print(header)
        print("-" * len(header))

        # Print metrics for each class
        for class_i in range(num_classes):
            metrics_values = [
                self.recall[epoch, class_i],
                self.TNR[epoch, class_i],
                self.FPR[epoch, class_i],
                self.PPV[epoch, class_i],
                self.NPV[epoch, class_i],
                self.f1[epoch, class_i],
                self.f2[epoch, class_i],
                self.g2[epoch, class_i],
                self.AP[epoch, class_i],
                self.loss[
                    epoch, class_i
                ],  # loss_metric ? why class_i ??? isn't self.loss stored as of epoch and of batch?
            ]
            row = f"{(class_i + 1):>6} | " + " | ".join(
                [f"{value:10.4f}" for value in metrics_values]
            )
            print("\n\n About to print a row on class_i = {class_i} \n\n")
            print(row)

        print("\n")

    def report_micro_averages(
        self, epoch, rewrite=False
    ):  # k is the k-fold CV fold (starts at 0 ends at 9?)
        # Define the metrics to report
        metrics = [
            "Recall",
            "TNR",
            "FPR",
            "PPV",
            "NPV",
            "F1",
            "F2",
            "Jaccard",
            "AP (micro)",
            "Challenge",
            "Loss",
        ]

        # Print the header
        header = "".join([f"{metric:>10}" for metric in metrics])
        print(header)
        print("-" * len(header))

        # Print metrics for each class
        metrics_values = [
            self.recall_micro[epoch],
            self.TNR_micro[epoch],
            self.FPR_micro[epoch],
            self.PPV_micro[epoch],
            self.NPV_micro[epoch],
            self.f1_micro[epoch],
            self.f2_micro[epoch],
            self.g2_micro[epoch],
            self.AP_micro[epoch],
            self.challenge_metric[epoch],
            self.loss_micro[epoch],
        ]

        row = "".join([f"{value:10.4f}" for value in metrics_values])
        print(row)

        print("\n")
        # Create a subfolder for each CV fold
        output_path_with_CV_fold = os.path.join(
            self.output_folder, f"{self.output_filename}"
        )

        if not os.path.exists(os.path.dirname(output_path_with_CV_fold)):
            os.makedirs(os.path.dirname(output_path_with_CV_fold))

        # Determine the file mode - overwrite if it's the first epoch and file exists
        file_mode = (
            "w"
            if epoch == 0
            and os.path.exists(output_path_with_CV_fold)
            or rewrite is True
            else "a"
        )

        # Append the metrics for the current epoch to the CSV file
        with open(output_path_with_CV_fold, file_mode, newline="") as csvfile:
            csvwriter = csv.writer(csvfile)

            # Write header if it's the first epoch
            if epoch == 0 or rewrite:
                csvwriter.writerow(["Epoch"] + metrics)

            # Write the metrics values
            csvwriter.writerow(
                [
                    f"{value:.4f}" if isinstance(value, float) else value
                    for value in [epoch] + metrics_values
                ]
            )

    def plot_loss(self):
        plt.figure(figsize=(10, 10))  # can change plot size
        plt.plot(np.mean(self.loss, axis=1), label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()
        plt.savefig(os.path.join(self.output_folder, f"loss_plot.png"))
        plt.close()
