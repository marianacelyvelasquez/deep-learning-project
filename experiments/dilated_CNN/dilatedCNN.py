import re
import os
import shutil
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

from utils.loss import BinaryFocalLoss
import utils.evaluate_12ECG_score as cinc_eval

from dataloaders.cinc2020.dataset import Cinc2020Dataset
from torch.utils.data import DataLoader
from models.dilated_CNN.model import CausalCNNEncoder as CausalCNNEncoderOld
from utils.MetricsManager import MetricsManager


class dilatedCNNExperiment:
    def __init__(self):
        self.network_params = {
            "in_channels": 12,
            "channels": 108,
            "depth": 6,
            "reduced_size": 216,
            "out_channels": 24,
            "kernel_size": 3,
        }

        self.device = self.get_device()
        self.model = self.load_model()

        # TODO: Currently we have one PyTorch Dataset in which we read all the d ata
        # and then we split it. Change it to first read all the data, then split it using
        # iterative stratification, then create PyTorch Datasets out of those.
        # Resampling and padding can be done in the Dataset class.
        (
            train_data,
            train_loader,
            validation_data,
            validation_loader,
            test_data,
            test_loader,
        ) = self.setup_datasets()

        self.train_dataset = train_data
        self.train_loader = train_loader

        self.validation_dataset = validation_data
        self.validation_loader = validation_loader

        self.test_dataset = test_data
        self.test_loader = test_loader

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = self.setup_loss_fn()

        self.num_epochs = 10

        self.train_metrics_manager = MetricsManager(
            num_epochs=self.num_epochs,
            num_classes=24,
            num_batches=len(self.train_loader),
            output="output/train_metrics.csv",
        )

        self.validation_metrics_manager = MetricsManager(
            num_epochs=self.num_epochs,
            num_classes=24,
            num_batches=len(self.validation_loader),
            output="output/validation_metrics.csv",
        )

        self.test_metrics_manager = MetricsManager(
            num_epochs=self.num_epochs,
            num_classes=24,
            num_batches=len(self.test_loader),
            output="output/test_metrics.csv",
        )

    def save_prediction(
        self, filenames, y_preds, y_probs, output_dir="output/predictions"
    ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Ensure y_trues and y_preds are numpy arrays for easy manipulation
        y_preds = y_preds.cpu().detach().numpy()
        y_probs = y_probs.cpu().detach().numpy()

        num_diagnoses = y_preds.shape[1]

        # Iterate over each sample
        for filename, y_pred, y_prob in zip(filenames, y_preds, y_probs):
            # Create a DataFrame for the current sampleo
            header = [f"diagnosis_{j+1}" for j in range(num_diagnoses)]
            df = pd.DataFrame(columns=header)
            df.loc[0] = y_pred
            df.loc[1] = y_prob

            # Save the DataFrame to a CSV file
            df.to_csv(os.path.join(output_dir, f"{filename}.csv"), index=False)

    def save_label(self, filenames, data_dir, output_dir, subdir):
        """
        Moves files from data_dir to a subdir in output_dir.

        Args:
        filenames (list of str): List of filenames without extensions.
        data_dir (str): Directory where the original files are located.
        output_dir (str): Base directory where the files will be moved.
        subdir (str): Subdirectory within output_dir where files will be moved.
        """
        # Ensure the output directory and subdirectory exist
        final_output_dir = os.path.join(output_dir, subdir)
        if not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir)

        # Move each file
        for filename in filenames:
            # Construct the full file paths
            original_path = os.path.join(data_dir, filename + ".hea")
            new_path = os.path.join(final_output_dir, filename + ".hea")

            # Move the file
            shutil.copy(original_path, new_path)

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

    def load_model(self):
        model = CausalCNNEncoderOld(**self.network_params)
        model = model.to(self.device)

        # Load pretrained model weights
        # TODO: Somehow doesn't properly work yet.
        freeze_modules = ["network\.0\.network\.[01234].*"]
        print("Freezing modules:", freeze_modules)
        exclude_modules = None

        # Load pretrained dilated CNN
        pretrained_cnn_weights = "models/dilated_CNN/pretrained_weights.pt"
        checkpoint = torch.load(pretrained_cnn_weights, map_location=self.device)
        checkpoint_dict = checkpoint["model_state_dict"]
        model_state_dict = model.state_dict()

        # filter out unnecessary keys
        # TODO: No sure what this is for i.e. why we just exlucde modules?
        # Was this simply for experimentation?
        if exclude_modules:
            checkpoint_dict = {
                k: v
                for k, v in checkpoint_dict.items()
                if k in model_state_dict
                and not any(re.compile(p).match(k) for p in exclude_modules)
            }

        # overwrite entries in the existing state dict
        model_state_dict.update(checkpoint_dict)

        # load the new state dict into the model
        model.load_state_dict(model_state_dict)

        # freeze the network's model weights of the module names
        # provided
        if not freeze_modules:
            return model
        for k, param in model.named_parameters():
            if any(re.compile(p).match(k) for p in freeze_modules):
                param.requires_grad = False

        return model

    def get_label_frequencies(self, loader):
        labels = []
        for _, _, label in loader:
            labels.append(label.numpy())

        labels = np.concatenate(labels, axis=0)

        freq = np.sum(labels, axis=0)

        return freq

    def calc_pos_weights(self, loader):
        freq = self.get_label_frequencies(loader)
        freq = np.where(freq == 0, freq.max(), freq)

        return torch.Tensor(np.around(freq.max() / freq, decimals=1))

    def setup_datasets(self):
        dataset = Cinc2020Dataset()

        # Fix randomness for reproducibility
        generator = torch.Generator().manual_seed(42)

        test_length = int(len(dataset) * 0.2)
        validation_length = int(len(dataset) * 0.1)
        train_length = len(dataset) - test_length - validation_length

        # Test, train, validation split
        train_data, test_data, validation_data = torch.utils.data.random_split(
            dataset, [train_length, test_length, validation_length], generator=generator
        )

        train_loader = DataLoader(train_data, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
        validation_loader = DataLoader(validation_data, batch_size=128, shuffle=False)

        train_freq = self.get_label_frequencies(train_loader) / len(train_loader)
        test_freq = self.get_label_frequencies(test_loader) / len(test_loader)
        validation_freq = self.get_label_frequencies(validation_loader) / len(
            validation_loader
        )

        # Labels for each list
        labels = ["Train freq.", "Test freq. ", "Valid freq."]

        # Function to print the lists as a table with labels
        print("\nLabel frequencies:")
        for label, lst in zip(labels, [train_freq, test_freq, validation_freq]):
            row = f"{label}: " + " ".join(f"{val:6.2f}" for val in lst)
            print(row)
        print("\n")

        return (
            train_data,
            train_loader,
            validation_data,
            validation_loader,
            test_data,
            test_loader,
        )

    def setup_loss_fn(self):
        # pos_weights = ((calc_pos_weights(loader) - 1) * .5) + 1
        # TODO: Make sure the values we compute make sense
        pos_weights = ((self.calc_pos_weights(self.train_loader) - 1) * 0.5) + 1

        loss_fn = BinaryFocalLoss(
            device=self.device,
            gamma=2,
            pos_weight=pos_weights,
        )

        return loss_fn

    def run(self):
        for epoch in range(self.num_epochs):
            # Train
            self.model.train()

            with tqdm(
                self.train_loader,
                desc=f"Training dilated CNN. Epoch {epoch +1}/{self.num_epochs}",
                total=len(self.train_loader),
            ) as pbar:
                for batch_i, (filenames, waveforms, labels) in enumerate(pbar):
                    self.optimizer.zero_grad()

                    waveforms = waveforms.float().to(self.device)
                    labels = labels.float().to(self.device)
                    # Note: The data is read as a float64 (double precision) array but the model expects float32 (single precision).
                    # So we have to convert it using .float()
                    predictions_logits = self.model(waveforms)

                    predictions_probabilities = torch.sigmoid(predictions_logits)
                    predictions = torch.round(predictions_probabilities)
                    loss = self.loss_fn(predictions, labels, self.model.training)

                    self.save_prediction(
                        filenames, predictions, predictions_probabilities
                    )
                    self.save_label(
                        filenames, "data/cinc2020_flattened", "output", "training"
                    )

                    loss.backward()
                    self.optimizer.step()

                    self.train_metrics_manager.update_confusion_matrix(
                        labels, predictions, epoch
                    )
                    self.train_metrics_manager.update_loss(loss, epoch, batch_i)

                    # Update tqdm postfix with current loss and accuracy
                    pbar.set_postfix(loss=f"\n{loss.item():.3f}")

                    # Update tqdm progress
                    pbar.update()

                self.train_metrics_manager.compute_micro_averages(epoch)
                self.train_metrics_manager.report_micro_averages(epoch)

                checkpoint_dir = "models/dilated_CNN/checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)

                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
                torch.save(self.model.state_dict(), checkpoint_path)

            # Validation evaluation
            self.model.eval()

            with tqdm(
                self.validation_loader,
                desc=f"Evaluate validation on dilated CNN. Epoch {epoch +1}",
                total=len(self.validation_loader),
            ) as pbar:
                # Train
                for batch_i, (filenames, waveforms, labels) in enumerate(pbar):
                    with torch.no_grad():
                        for batch_i, (filenames, waveforms, labels) in enumerate(
                            self.validation_loader
                        ):
                            waveforms = waveforms.float().to(self.device)
                            labels = labels.float().to(self.device)

                            predictions = self.model(waveforms)
                            predictions = torch.sigmoid(predictions)
                            predictions = torch.round(predictions)

                            loss = self.loss_fn(
                                predictions, labels, self.model.training
                            )
                            self.validation_metrics_manager.update_loss(
                                loss, epoch, batch_i
                            )
                            self.validation_metrics_manager.update_confusion_matrix(
                                labels, predictions, epoch
                            )

                            self.save_prediction(filenames, labels, predictions)
                            self.save_label(
                                filenames,
                                "data/cinc2020_flattened",
                                "output",
                                "validation",
                            )

                self.validation_metrics_manager.compute_micro_averages(epoch)
                self.validation_metrics_manager.report_micro_averages(epoch)

            # TEST set evaluation
            with tqdm(
                self.test_loader,
                desc=f"Evaluate test data on dilated CNN. Epoch {epoch +1}",
                total=len(self.test_loader),
            ) as pbar:
                # Train
                for batch_i, (filenames, waveforms, labels) in enumerate(pbar):
                    with torch.no_grad():
                        for batch_i, (filenames, waveforms, labels) in enumerate(
                            self.validation_loader
                        ):
                            waveforms = waveforms.float().to(self.device)
                            labels = labels.float().to(self.device)

                            predictions = self.model(waveforms)
                            predictions = torch.sigmoid(predictions)
                            predictions = torch.round(predictions)

                            loss = self.loss_fn(
                                predictions, labels, self.model.training
                            )
                            self.test_metrics_manager.update_loss(loss, epoch, batch_i)
                            self.test_metrics_manager.update_confusion_matrix(
                                labels, predictions, epoch
                            )

                            self.save_prediction(filenames, labels, predictions)
                            self.save_label(
                                filenames, "data/cinc2020_flattened", "output", "test"
                            )

                self.test_metrics_manager.compute_micro_averages(epoch)
                self.test_metrics_manager.report_micro_averages(epoch)

        print(
            "Run the challenges e valuation code e.g.: python utils/evaluate_12ECG_score.py output/training output/predictions"
        )
