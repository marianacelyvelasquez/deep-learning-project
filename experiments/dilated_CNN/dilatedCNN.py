import re
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import csv
import torch

from experiments.dilated_CNN.config import Config

from dataloaders.cinc2020.datasetThePaperCode import Cinc2020Dataset
from torch.utils.data import DataLoader
from utils.MetricsManager import MetricsManager
from dataloaders.cinc2020.common import labels_map
from utils.load_optimizer import load_optimizer
from utils.setup_loss_fn import setup_loss_fn

class dilatedCNNExperiment:
    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        classes,
        classes_test,
        CV_k,
        checkpoint_path=None,
    ):
        torch.manual_seed(42)

        self.network_params = {
            "in_channels": 12,
            "channels": 108,
            "depth": 6,
            "reduced_size": 216,
            "out_channels": 24,
            "kernel_size": 3,
        }

        # Load pretrained dilated CNN
        if checkpoint_path is None:
            pretrained_cnn_weights = "models/dilated_CNN/pretrained_weights.pt"
            self.checkpoint_path = pretrained_cnn_weights
        else:
            self.checkpoint_path = checkpoint_path

        self.CV_k = CV_k

        self.device = self.get_device()

        self.classes = classes
        self.classes_test = classes_test

        # TODO: Currently we have one PyTorch Dataset in which we read all the d ata
        # and then we split it. Change it to first read all the data, then split it using
        # iterative stratification, then create PyTorch Datasets out of those.
        # Resampling and padding can be done in the Dataset class.
        # (
        #     train_loader,
        #     validation_loader,
        #     test_loader,
        # ) = self.setup_datasets()

        # Create datasets

        # THIS IS FOR DATASET THEIR NOT OUR OUR IS A BIT DIFFERENT TO CALL
        self.train_dataset = Cinc2020Dataset(
            X_train,
            y_train,
            classes=classes,
            root_dir=Config.TRAIN_DATA_DIR,
            name="train",
        )
        self.validation_dataset = Cinc2020Dataset(
            X_val, y_val, classes=classes, root_dir=Config.TRAIN_DATA_DIR, name="val"
        )
        self.test_dataset = Cinc2020Dataset(
            X_test, y_test, classes=classes_test, root_dir=Config.TEST_DATA_DIR, name="test"
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)
        self.validation_loader = DataLoader(
            self.validation_dataset, batch_size=128, shuffle=True
        )

        self.model = self.load_model()

        self.optimizer = load_optimizer(self.model, self.device, self.checkpoint_path, load_optimizer=Config.LOAD_OPTIMIZER, learning_rate=Config.LEARNING_RATE)
        self.loss_fn = setup_loss_fn(self.train_loader, self.device) #you only pass the loader to compute the positive weights, which are only needed when training

        self.min_num_epochs = Config.MIN_NUM_EPOCHS
        self.max_num_epochs = Config.MAX_NUM_EPOCHS
        self.epoch = 0

        self.num_classes = 24

        self.predictions = []

        self.train_metrics_manager = MetricsManager(
            name="train",
            num_epochs=self.max_num_epochs,
            num_classes=self.num_classes,
            num_batches=len(self.train_loader),
            CV_k=self.CV_k,
            classes=self.classes,
        )

        self.validation_metrics_manager = MetricsManager(
            name="validation",
            num_epochs=self.max_num_epochs,
            num_classes=self.num_classes,
            num_batches=len(self.validation_loader),
            CV_k=self.CV_k,
            classes=self.classes,
        )

        self.test_metrics_manager = MetricsManager(
            name="test",
            num_epochs=self.max_num_epochs,
            num_classes=self.num_classes,
            num_batches=len(self.test_loader),
            CV_k=self.CV_k,
            classes=self.classes_test,
        )

    def save_prediction(self, filenames, y_preds, y_probs, subdir):
        output_dir = os.path.join(
            Config.OUTPUT_DIR, f"fold_{self.CV_k}", "predictions", subdir
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

    def save_label(self, filenames):
        """
        Moves files from data_dir to a subdir in output_dir.

        Args:
        filenames (list of str): List of filenames without extensions.
        data_dir (str): Directory where the original files are located.
        output_dir (str): Base directory where the files will be moved.
        subdir (str): Subdirectory within output_dir where files will be moved.
        """
        # Get the absolute path of the current directory
        current_dir = os.getcwd()

        # Ensure the output directory and subdirectory exist
        output_dir = os.path.join(
            current_dir, Config.OUTPUT_DIR, f"fold_{self.CV_k}", "train_data"
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Move each file
        for filename in filenames:
            # Construct the full file paths
            original_path = os.path.join(current_dir, Config.TRAIN_DATA_DIR, filename)
            target_path = os.path.join(current_dir, output_dir, filename)

            # Move the file
            os.symlink(original_path + ".hea", target_path + ".hea")
            os.symlink(original_path + ".mat", target_path + ".mat")

    def get_device(self):
        # Check if CUDA is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA")
        # Check if MPS is available
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using METAL GPU")
        # If neither CUDA nor MPS is available, use CPU
        else:
            device = torch.device("cpu")
            print("Using CPU")

        return device

    def read_records(self, source_dir):
        records = []
        for dirpath, dirnames, filenames in os.walk(source_dir):
            # Read all records from a current g* subdirectory.
            for filename in filenames:
                if filename.endswith(".hea"):
                    record_path = os.path.join(dirpath, filename.split(".")[0])
                    records.append(record_path)

        return records

    def run_epochs(self):
        print("Storing training labels in output folder for later evlauation.")
        for batch_i, (filenames, waveforms, labels) in enumerate(self.train_loader):
            # We save a symlink of the training set to the output dir.
            # We do that because the evaluation script expects to find
            # for every file in the input directory a corresponding
            # prediction file. Isusue is: We do CV, so the training set
            # we get here is a subset of the one in the data dir.
            # So we have to record it somehow.

            self.save_label(filenames)

        for epoch in range(self.max_num_epochs):
            self.epoch = epoch

            # Train
            self.model.train()

            with tqdm(
                self.train_loader,
                desc=f"\033[32mTraining dilated CNN. Epoch {epoch +1}/{self.max_num_epochs}\033[0m",
                total=len(self.train_loader),
            ) as pbar:
                # Reset predictions
                self.predictions = []

                # Reset labels
                self.labels = []

                for batch_i, (filenames, waveforms, labels) in enumerate(pbar):
                    self.optimizer.zero_grad()

                    # Store labels, this is needed to capture the order
                    self.labels.extend(labels)

                    # Note: The data is read as a float64 (double precision) array but the model expects float32 (single precision).
                    # So we have to convert it using .float()
                    waveforms = waveforms.float().to(self.device)
                    labels = labels.float().to(self.device)

                    predictions_logits = self.model(waveforms)

                    # We compute the loss on the logits, not the probabilities
                    loss = self.loss_fn(
                        predictions_logits, labels, self.model.training
                    )

                    # Convert logits to probabilities and round to get predictions
                    predictions_probabilities = torch.sigmoid(predictions_logits)
                    predictions = torch.round(predictions_probabilities)

                    self.predictions.extend(predictions.cpu().detach().tolist())

                    self.save_prediction(
                        filenames, predictions, predictions_probabilities, "train"
                    )

                    loss.backward()
                    self.optimizer.step()

                    # Metrics are updated once per batch (batch size here by default is 128)
                    self.train_metrics_manager.update_confusion_matrix(
                        labels, predictions, epoch
                    )
                    self.train_metrics_manager.update_loss(loss, epoch, batch_i)

                    # Update tqdm postfix with current loss and accuracy
                    pbar.set_postfix(loss=f"\n{loss.item():.3f}")

                    # Update tqdm progress
                    pbar.update()

                self.train_metrics_manager.compute_macro_averages(epoch)
                self.train_metrics_manager.compute_challenge_metric(
                    self.labels, self.predictions, epoch
                )
                self.train_metrics_manager.report_macro_averages(epoch)

                checkpoint_dir = os.path.join(
                    Config.OUTPUT_DIR, f"fold_{self.CV_k}", "checkpoints"
                )
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir, exist_ok=True)

                checkpoint_file_path = os.path.join(
                    checkpoint_dir, f"epoch_{epoch + 1}.pt"
                )

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

            # Validation evaluation
            self.model.eval()

            with tqdm(
                self.validation_loader,
                desc=f"\033[34mEvaluate validation on dilated CNN. Epoch {epoch +1}\033[0m",
                total=len(self.validation_loader),
            ) as pbar:
                # Reset predictions
                self.predictions = []

                # Reset labels store
                self.labels = []

                # Validation
                with torch.no_grad():
                    for batch_i, (filenames, waveforms, labels) in enumerate(pbar):
                        # Store labels, this is needed to capture the order
                        self.labels.extend(labels)

                        # Note: The data is read as a float64 (double precision) array but the model expects float32 (single precision).
                        # So we have to convert it using .float()
                        waveforms = waveforms.float().to(self.device)
                        labels = labels.float().to(self.device)

                        predictions_logits = self.model(waveforms)

                        # We compute the loss on the logits, not the probabilities
                        loss = self.train_loss_fn(
                            predictions_logits, labels, self.model.training
                        )

                        # Convert logits to probabilities and round to get predictions
                        predictions_probabilities = torch.sigmoid(predictions_logits)
                        predictions = torch.round(predictions_probabilities)

                        self.predictions.extend(predictions.cpu().detach().tolist())

                        self.validation_metrics_manager.update_loss(
                            loss, epoch, batch_i
                        )
                        self.validation_metrics_manager.update_confusion_matrix(
                            labels, predictions, epoch
                        )

                        # self.save_prediction(
                        #     filenames,
                        #     predictions,
                        #     predictions_probabilities,
                        #     "validation",
                        # )

                        # TODO: Dont pass config, just use it inside the function.
                        # Do it for all save_label calls.
                        # self.save_label(
                        #     filenames,
                        #     Config.DATA_DIR,
                        #     Config.OUTPUT_DIR,
                        #     "validation",
                        # )

                self.validation_metrics_manager.compute_macro_averages(epoch)
                self.validation_metrics_manager.compute_challenge_metric(
                    self.labels, self.predictions, epoch
                )
                self.validation_metrics_manager.report_macro_averages(epoch)

            stop_early = self.validation_metrics_manager.check_early_stopping()
            if stop_early is True:
                print("\033[91mstopping early\033[0m")
                break

        self.train_metrics_manager.plot_loss()

        self.evaluate_test_set()

    def evaluate_test_set(self):
        labels_stored = []

        # TEST set evaluation
        self.model.eval()
        with tqdm(
            self.test_loader,
            desc="\033[33mEvaluate test data on dilated CNN.\033[0m",
            total=len(self.test_loader),
        ) as pbar:
            # Reset predictions
            self.preds = []
            self.preds_probs = []

            self.labels = []

            # Test
            with torch.no_grad():
                for batch_i, (filenames, waveforms, labels) in enumerate(pbar):
                    # last_epoch = self.num_epochs - 1

                    # Note: The data is read as a float64 (double precision) array but the model expects float32 (single precision).
                    # So we have to convert it using .float()
                    waveforms = waveforms.float().to(self.device)
                    labels = labels.float().to(self.device)

                    predictions_logits = self.model(waveforms)

                    # We compute the loss on the logits, not the probabilities
                    loss = self.loss_fn(
                        predictions_logits, labels, self.model.training
                    )

                    # Convert logits to probabilities and round to get predictions
                    predictions_probabilities = torch.sigmoid(predictions_logits)
                    self.preds_probs.extend(
                        predictions_probabilities.cpu().detach().tolist()
                    )
                    predictions = torch.round(predictions_probabilities)

                    self.predictions.extend(predictions.cpu().detach().tolist())
                    self.labels.extend(labels.cpu().detach().tolist())

                    self.test_metrics_manager.update_loss(
                        loss, epoch=0, batch_i=batch_i
                    )
                    self.test_metrics_manager.update_confusion_matrix(
                        labels, predictions, epoch=0
                    )

                    self.save_prediction(
                        filenames, predictions, predictions_probabilities, "test"
                    )
                    # self.save_label(
                    #     filenames, Config.DATA_DIR, Config.OUTPUT_DIR, "test"
                    # )

            self.test_metrics_manager.compute_macro_averages(epoch=0)
            self.test_metrics_manager.compute_challenge_metric(
                self.labels, self.predictions, epoch=0
            )

            self.test_metrics_manager.report_macro_averages(epoch=0, rewrite=True)

        print(
            f"Run the challenges e valuation code e.g.: python utils/evaluate_12ECG_score.py output/training output/predictions for CV-fold k={self.CV_k}"
        )
