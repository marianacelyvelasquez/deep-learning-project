import torch
import collections
from tqdm import tqdm
import math
import os
import csv
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.MetricsManager import MetricsManager

# from dataloaders.cinc2020.dataset import Cinc2020Dataset
from dataloaders.cinc2020.datasetThePaperCode import Cinc2020Dataset
from utils.load_model import load_model
from utils.setup_loss_fn import setup_loss_fn
from utils.get_device import get_device
from utils.load_optimizer import load_optimizer
from dataloaders.cinc2020.common import labels_map


class SWAGInference:
    def __init__(
        self,
        checkpoint_path,  # end: for dilatedCNN
        X_train,
        y_train,
        X_test,
        y_test,
        classes,
        classes_test,
        Config,
        CV_k,
    ) -> None:
        # Fix randomness
        torch.manual_seed(42)

        # Define num swag max epochs, swag LR, swag update freq, deviation matrix max rank, num BMA samples
        self.swag_max_epochs = Config.MAX_NUM_EPOCHS
        self.swag_learning_rate = Config.SWAG_LEARNING_RATE
        self.swag_update_freq = Config.SWAG_UPDATE_FREQ
        self.deviation_matrix_max_rank = Config.DEVIATION_MATRIX_MAX_RANK
        self.bma_samples = Config.BMA_SAMPLES
        self.output_dir = Config.OUTPUT_DIR


        # Define classes
        self.classes = classes
        self.classes_test = classes_test

        ##
        # Load training data, test data
        ##
        # Create datasets
        self.train_dataset = Cinc2020Dataset(
            X_train,
            y_train,
            classes=self.classes,
            root_dir=Config.TRAIN_DATA_DIR,
            name="train",
        )
        self.test_dataset = Cinc2020Dataset(
            X_test,
            y_test,
            classes=self.classes_test,
            root_dir=Config.TEST_DATA_DIR,
            name="test",
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)

        ##
        # Load model
        ##
        self.network_params = {
            "in_channels": 12,
            "channels": 108,
            "depth": 6,
            "reduced_size": 216,
            "out_channels": 24,
            "kernel_size": 3,
        }

        device_type, device_count = get_device()
        self.device = torch.device(f"{device_type}:0")

        # Load pretrained dilated CNN
        if checkpoint_path is None:
            pretrained_cnn_weights = "models/dilated_CNN/pretrained_weights.pt"
            self.checkpoint_path = pretrained_cnn_weights
        else:
            self.checkpoint_path = checkpoint_path

        self.model = load_model(self.network_params, self.device, self.checkpoint_path)

        # Define optimizer and learning rate scheduler
        self.optimizer = load_optimizer(
            self.model,
            self.device,
            self.checkpoint_path,
            load_optimizer=Config.LOAD_OPTIMIZER,
            learning_rate=self.swag_learning_rate,
        )

        # Define loss
        self.loss_fn = setup_loss_fn(self.train_loader, self.device)

        # Allocate memory for theta, theta_squared, D (D-matrix)
        self.theta = self._create_weight_copy()
        self.theta_squared = self._create_weight_copy()

        self.D = collections.deque(maxlen=self.deviation_matrix_max_rank)

        # Define prediction threshold
        self.prediction_threshold = 0.5

        # set current iteration n - used to compute sigma etc.
        self.n = 0

        ##
        # Metrics
        ##

        self.CV_k = CV_k

        self.num_classes = 24

        self.predictions = []

        self.train_metrics_manager = MetricsManager(
            name="train",
            num_classes=self.num_classes,
            num_batches=len(self.train_loader),
            CV_k=self.CV_k,
            classes=self.classes,
            Config=Config,
        )

        self.test_metrics_manager = MetricsManager(
            name="test",
            num_classes=self.num_classes,
            num_batches=len(self.test_loader),
            CV_k=self.CV_k,
            classes=self.classes_test,
            Config=Config,
        )

    def update_swag(self) -> None:
        current_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }
        # print dictionary current_params:
        print(f"Current params items length: {len(current_params.items())}")

        # Update swag diagonal
        for name, param in current_params.items():
            self.theta[name] = (self.n * self.theta[name] + param) / (self.n + 1)
            self.theta_squared[name] = (
                self.n * self.theta_squared[name] + param**2
            ) / (self.n + 1)

        # Update full swag
        theta_difference = self._create_weight_copy()
        for name, param in current_params.items():
            theta_difference[name] = torch.clamp(param - self.theta[name], min=1e-10)
        self.D.append(theta_difference)

    def fit_swag(self) -> None:
        self.theta = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }
        self.theta_squared = {
            name: param.detach().clone() ** 2
            for name, param in self.model.named_parameters()
        }

        for epoch in range(self.swag_max_epochs):
            print(f"Epoch number: {epoch}")
            self.D.clear()  # clear D-matrix

            self.model.train()
            with tqdm(
                self.train_loader,
                desc=f"\033[32mTraining SWAG. Epoch {epoch}/{self.swag_max_epochs}\033[0m",
                total=len(self.train_loader),
            ) as pbar:
                # Reset predictions
                self.predictions = []
                labels_for_epoch = []

                for batch_i, (filenames, waveforms, labels) in enumerate(pbar):
                    self.optimizer.zero_grad()
                    labels_for_epoch.extend(labels)

                    # Note: The data is read as a float64 (double precision) array but the model expects float32 (single precision).
                    # So we have to convert it using .float()
                    waveforms = waveforms.float().to(self.device)
                    labels = labels.float().to(self.device)

                    predictions_logits = self.model(waveforms)
                    assert predictions_logits.shape == labels.shape

                    # We compute the loss on the logits, not the probabilities
                    loss = self.loss_fn(predictions_logits, labels, self.model.training)

                    # Convert logits to probabilities and round to get predictions
                    predictions_probabilities = torch.sigmoid(predictions_logits)
                    predictions = torch.round(predictions_probabilities)

                    self.predictions.extend(predictions.cpu().detach().tolist())

                    self.save_prediction(
                        filenames, predictions, predictions_probabilities, "train"
                    )

                    loss.backward()  # question: why is this .backward fn not recognized # check that his actually works
                    self.optimizer.step()

                    ##
                    # Metrics are updated once per batch (batch size here by default is 128)
                    ##
                    self.train_metrics_manager.update_confusion_matrix(
                        labels, predictions, epoch
                    )
                    self.train_metrics_manager.update_loss(loss, epoch, batch_i)

                    # Update tqdm postfix with current loss and accuracy
                    pbar.set_postfix(loss=f"\n{loss.item():.3f}")
                    pbar.update()

                    ## End of just updating metrics

                if epoch % self.swag_update_freq == 0:
                    self.n = (epoch + 1) // self.swag_update_freq
                    self.update_swag()

            ### NEW STUFF
            labels_for_epoch_numpy = [np.array(y_elem) for y_elem in labels_for_epoch]

            self.train_metrics_manager.compute_macro_averages(epoch)
            self.train_metrics_manager.compute_challenge_metric(
                labels_for_epoch_numpy, self.predictions, epoch
            )
            self.train_metrics_manager.report_macro_averages(epoch)

            checkpoint_dir = os.path.join(
                self.output_dir, f"fold_{self.CV_k}", "checkpoints"
            )
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_file_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")

            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "network_name": "dialted_CNN",
                    "network_params": self.network_params,
                },
                checkpoint_file_path,
            )

    def predict_probabilities(
        self, loader: torch.utils.data.DataLoader
    ):
        """
        Goal: Implement Bayesian Model Averaging by doing:
             1. Sample new model from SWAG (call self.sample_parameters())
             2. Predict probabilities with sampled model
             3. repeat 1-2 for num_bma_samples times
             4. Average the probabilities over the num_bma_samples
        """

        with torch.no_grad():
            self.model.eval()

            per_model_sample_predictions = []
            for bma_i in tqdm(range(self.bma_samples), desc="Bayesian Model Averaging"):
                self.sample_parameters()
                predictions = self._predict_probabilities_of_model(loader)
                per_model_sample_predictions.append(predictions)

            bma_probabilities = torch.mean(
                torch.stack(per_model_sample_predictions, dim=0), dim=0
            )
            return bma_probabilities

    def sample_parameters(self) -> None:
        # Note: Be sure to do the math here correctly!
        # The solution might have a tiny error in it.

        # Instead of acting on a full vector of parameters, all operations can be done on per-layer parameters.

        for name, param in self.model.named_parameters():
            # SWAG-diagonal part
            z_1 = torch.randn(param.size()).to(self.device)

            current_mean = self.theta[name]
            current_std = torch.sqrt(
                torch.clamp(self.theta_squared[name] - current_mean**2, min=1e-10)
            )

            assert (
                current_mean.size() == param.size()
                and current_std.size() == param.size()
            )

            sampled_param = current_mean + (1.0 / math.sqrt(2.0)) * current_std * z_1

            z_2 = torch.randn(param.size()).to(self.device)

            for D_i in self.D:
                sampled_param += (
                    1.0 / math.sqrt(2 * self.deviation_matrix_max_rank - 1)
                ) * torch.clamp(D_i[name] * z_2, min=1e-10)

            param.data = sampled_param

        self._update_batchnorm()

    def _create_weight_copy(self) -> None:
        """Create an all-zero copy of the network weights as a dictionary that maps name -> weight (matrix)"""
        return {
            name: torch.zeros_like(param, requires_grad=False)
            for name, param in self.model.named_parameters()
        }

    def _predict_probabilities_of_model(
        self, loader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        predictions = []
        for _, waveforms, _ in loader:
            waveforms = waveforms.float().to(self.device)
            predictions.append(self.model(waveforms))

        predictions = torch.cat(predictions)
        return F.sigmoid(predictions)

    def _update_batchnorm(self) -> None:
        """
        Reset and fit batch normalization statistics using the training dataset self.train_dataset.
        See the SWAG paper for why this is required.

        Batch normalization usually uses an exponential moving average, controlled by the `momentum` parameter.
        However, we are not training but want the statistics for the full training dataset.
        Hence, setting `momentum` to `None` tracks a cumulative average instead.
        The following code stores original `momentum` values, sets all to `None`,
        and restores the previous hyperparameters after updating batchnorm statistics.
        """

        old_momentum_parameters = dict()
        for module in self.model.modules():
            # Only need to handle batchnorm modules
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue

            # Store old momentum value before removing it
            old_momentum_parameters[module] = module.momentum
            module.momentum = None

            # Reset batch normalization statistics
            #  :eyes: reset the running statistics of the batch normalization layer,
            #  including mean and variance. Done to compute fresh statistics
            #  using the entire training dataset during the inference phase
            module.reset_running_stats()

        self.model.train()

        # Only done for test set PROBABLY TODO: check /!\
        for _, waveforms, _ in self.test_loader:
            waveforms = waveforms.float().to(self.device)
            self.model(waveforms)
        self.model.eval()

        # Restore old `momentum` hyperparameter values
        for module, momentum in old_momentum_parameters.items():
            module.momentum = momentum

    def evaluate(self) -> None:
        self.model.eval()

        loader = self.test_loader

        predicted_test_probabilities = self.predict_probabilities(loader)

        # Calibration bins
        bins = np.linspace(0, 1, 11)
        bin_counts = {
            label: np.zeros(len(bins) - 1) for label in range(len(labels_map))
        }
        bin_sums = {label: np.zeros(len(bins) - 1) for label in range(len(labels_map))}

        batch_size = loader.batch_size
        num_batches = len(loader)
        remainder = predicted_test_probabilities.size(0) % batch_size

        assert len(loader.dataset) == predicted_test_probabilities.size(
            0
        ), "Size mismatch between loader and logits"

        with tqdm(
            loader,
            desc="\033[33mEvaluating test data with SWAGInference.\033[0m",
            total=num_batches,
        ) as pbar:
            labels_total = []
            for (batch_i, (filenames, _, labels)), probabilities_chunk in zip(
                enumerate(pbar),
                torch.split(predicted_test_probabilities, batch_size, dim=0),
            ):
                labels_total.extend(labels)
                if batch_i == num_batches - 1 and remainder != 0:
                    probabilities_chunk = predicted_test_probabilities[-remainder:]

                labels = labels.float().to(self.device)
                predicted_test_logits_chunk = torch.logit(probabilities_chunk)

                loss = self.loss_fn(
                    predicted_test_logits_chunk, labels, self.model.training
                )

                # this means threshold = 0.5
                predictions = torch.round(probabilities_chunk)
                self.predictions.extend(predictions.cpu().detach().tolist())

                self.test_metrics_manager.update_loss(loss, epoch=0, batch_i=batch_i)
                self.test_metrics_manager.update_confusion_matrix(
                    labels, predictions, epoch=0
                )

                # Save predictions if needed
                self.save_prediction(
                    filenames, predictions, probabilities_chunk, "test"
                )

                # Assign probabilities to bins and count occurrences
                for label in range(len(labels_map)):
                    probs = probabilities_chunk[:, label]
                    actuals = labels[:, label]
                    for b in range(len(bins) - 1):
                        in_bin = (probs >= bins[b]) & (probs < bins[b + 1])
                        bin_counts[label][b] += in_bin.sum().item()
                        bin_sums[label][b] += actuals[in_bin].sum().item()

        # Calculate and save calibration data
        calibration_data = {}
        for label in range(len(labels_map)):
            calibration_data[labels_map[label]] = {
                "bin": [],
                "predicted_probability": [],
                "actual_frequency": [],
                "absolute_occurrences": [],
                "total_samples_in_bin": [],
            }
            for b in range(len(bins) - 1):
                pred_prob = (bins[b] + bins[b + 1]) / 2
                calibration_data[labels_map[label]]["bin"].append(
                    b + 1
                )  # Numeric bin labeling
                calibration_data[labels_map[label]]["predicted_probability"].append(
                    pred_prob
                )

                # Handle bins with zero occurrences
                if bin_counts[label][b] > 0:
                    actual_freq = bin_sums[label][b] / bin_counts[label][b]
                else:
                    actual_freq = 0

                calibration_data[labels_map[label]]["actual_frequency"].append(
                    actual_freq
                )
                calibration_data[labels_map[label]]["absolute_occurrences"].append(
                    bin_sums[label][b]
                )
                calibration_data[labels_map[label]]["total_samples_in_bin"].append(
                    bin_counts[label][b]
                )

        # Export calibration data to CSV
        for label, data in calibration_data.items():
            output_path = os.path.join(
                self.output_dir,
                "fold_0",
                "metrics",
                "calibration_measurements",
                f"calibration_{label}.csv",
            )

            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))

            with open(output_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=[
                        "bin",
                        "predicted_probability",
                        "actual_frequency",
                        "absolute_occurrences",
                        "total_samples_in_bin",
                    ],
                )
                writer.writeheader()
                for i in range(len(data["bin"])):
                    writer.writerow({k: data[k][i] for k in data})

        self.test_metrics_manager.compute_macro_averages(epoch=0)
        self.test_metrics_manager.compute_challenge_metric(
            labels_total, self.predictions, epoch=0
        )
        self.test_metrics_manager.report_macro_averages(epoch=0, rewrite=True)

    def save_prediction(self, filenames, y_preds, y_probs, subdir):
        output_dir = os.path.join(
            self.output_dir, f"fold_{self.CV_k}", "predictions", subdir
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Ensure y_trues and y_preds are numpy arrays for easy manipulation
        y_preds = y_preds.cpu().detach().numpy().astype(int)
        y_probs = y_probs.cpu().detach().numpy().round(decimals=2)

        # Iterate over each sample
        for filename, y_pred, y_prob in zip(filenames, y_preds, y_probs):
            # Create and open a CSV file for the current sample
            with open(
                os.path.join(output_dir, f"{filename}.csv"), mode="w", newline=""
            ) as file:
                writer = csv.writer(file)
                # Write header
                header = labels_map.tolist()
                writer.writerow(header)
                # Write y_pred and y_prob as separate rows
                writer.writerow(y_pred)
                writer.writerow(y_prob)
