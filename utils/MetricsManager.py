import numpy as np
import numpy.typing as npt
from sklearn.metrics import multilabel_confusion_matrix


class MetricsManager:
    def __init__(self, num_epochs, num_classes, num_batches):
        # True Positives
        self.tp: npt.NDArray[np.int_] = np.zeros((num_epochs, num_classes))
        # False Negatives
        self.fn: npt.NDArray[np.int_] = np.zeros((num_epochs, num_classes))
        # True Positives(
        self.fp: npt.NDArray[np.int_] = np.zeros((num_epochs, num_classes))
        # True Negatives
        self.tn: npt.NDArray[np.int_] = np.zeros((num_epochs, num_classes))

        self.loss: npt.NDArray[np.float_] = np.zeros((num_epochs, num_batches))

        # TODO: Implement
        self.challenge_metric: npt.NDArray[np.float_] = np.zeros(
            (num_epochs, num_classes)
        )
        self.sensitivity: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        self.specificity: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        self.FPR: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        self.PPV: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        self.NPV: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        self.f1: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        self.f2: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        self.g2: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        self.ROCAUC: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))
        self.AP: npt.NDArray[np.float_] = np.zeros((num_epochs, num_classes))

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

    def update_loss(self, loss, epoch, batch_i):
        self.loss[epoch][batch_i] = loss.detach().cpu()

    def compute_metrics(self, epoch):
        self.sensitivity[epoch] = self.compute_sensitivity(epoch)
        self.specificity[epoch] = self.compute_specificity(epoch)
        self.FPR[epoch] = self.compute_false_positive_rate(epoch)
        self.PPV[epoch] = self.compute_positive_predictive_value(epoch)
        self.NPV[epoch] = self.compute_negative_predictive_value(epoch)
        self.f1[epoch] = self.compute_f1(epoch)
        self.f2[epoch] = self.compute_fbeta(epoch)
        self.g2[epoch] = self.compute_jaccard(epoch)
        # TODO: Implement average precision score

        print("Updated all metrics for the current epoch.")

    def compute_sensitivity(self, epoch):
        # Calculate sensitivity for each class
        sensitivity = self.tp[epoch] / (self.tp[epoch] + self.fn[epoch])
        sensitivity = np.nan_to_num(sensitivity)

        return sensitivity

    def compute_specificity(self, epoch):
        specificity = self.tn[epoch] / (self.tn[epoch] + self.fp[epoch])
        specificity = np.nan_to_num(specificity)

        return specificity

    def compute_false_positive_rate(self, epoch):
        false_positive_rate = self.fp[epoch] / (self.tn[epoch] + self.fp[epoch])
        false_positive_rate = np.nan_to_num(false_positive_rate)

        return false_positive_rate

    def compute_positive_predictive_value(self, epoch):
        positive_predictive_value = self.tp[epoch] / (self.tp[epoch] + self.fp[epoch])
        positive_predictive_value = np.nan_to_num(positive_predictive_value)

        return positive_predictive_value

    def compute_negative_predictive_value(self, epoch):
        negative_predictive_value = self.tn[epoch] / (self.tn[epoch] + self.fn[epoch])
        negative_predictive_value = np.nan_to_num(negative_predictive_value)

        return negative_predictive_value

    def compute_f1(self, epoch):
        recall = self.sensitivity[epoch]
        precision = self.PPV[epoch]
        f1 = 2 * ((recall * precision) / (recall + precision))
        f1 = np.nan_to_num(f1)

        return f1

    def compute_fbeta(self, epoch, beta=2):
        fbeta = ((1 + beta**2) * self.tp[epoch]) / (
            (1 + beta**2) * self.tp[epoch] + self.fp[epoch] + (beta**2) * self.fn[epoch]
        )
        fbeta = np.nan_to_num(fbeta)

        return fbeta

    def compute_jaccard(self, epoch, beta=2):
        jaccard = self.tp[epoch] / (
            self.tp[epoch] + self.fp[epoch] + beta * self.fn[epoch]
        )
        jaccard = np.nan_to_num(jaccard)

        return jaccard

    def report(self, epoch):
        # Define the metrics to report
        metrics = [
            "Sensitivity",
            "Specificity",
            "FPR",
            "PPV",
            "NPV",
            "F1",
            "F2",
            "Jaccard",
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
                self.sensitivity[epoch, class_i],
                self.specificity[epoch, class_i],
                self.FPR[epoch, class_i],
                self.PPV[epoch, class_i],
                self.NPV[epoch, class_i],
                self.f1[epoch, class_i],
                self.f2[epoch, class_i],
                self.g2[epoch, class_i],
            ]
            row = f"{(class_i + 1):>6} | " + " | ".join(
                [f"{value:10.4f}" for value in metrics_values]
            )
            print(row)

        print("\n")
