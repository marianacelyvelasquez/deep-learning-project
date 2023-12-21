import numpy as np
import numpy.typing as npt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix


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
        self.challenge_metric = None
        self.sensitivity = None
        self.specificity = None
        self.FPR = None
        self.PPV = None
        self.NPV = None
        self.F1 = None
        self.F2 = None
        self.G2 = None
        self.ROCAUC = None
        self.AP = None

    def update(self, y_trues, y_preds, loss, epoch, batch):
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
        loss = loss.detach().cpu()

        self.loss[epoch][batch] = loss

        # Format: https://scikit-learn.org/stable/glossary.html#term-multilabel-indicator-matrices
        # Gives us a list of confusion matrices for each class and for the whole batch
        confusion_matrices = multilabel_confusion_matrix(y_trues, y_preds)

        # TODO: Check computation of tp, fn etc. manually again and make sure it's correct 100%
        for confusion_matrix in confusion_matrices:
            # Compute confusion matrix for each label in the current sample
            for class_i, confusion_matrix in enumerate(confusion_matrices):
                tn, fp, fn, tp = confusion_matrix.ravel()

                self.tp[epoch][class_i] += tp
                self.fn[epoch][class_i] += fn
                self.fp[epoch][class_i] += fp
                self.tn[epoch][class_i] += tn
