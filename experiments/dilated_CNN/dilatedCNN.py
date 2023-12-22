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

network_params = {
    "in_channels": 12,
    "channels": 108,
    "depth": 6,
    "reduced_size": 216,
    "out_channels": 24,
    "kernel_size": 3,
}


def save_prediction(filenames, y_preds, y_probs, output_dir="output/predictions"):
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


def save_label(filenames, data_dir, output_dir, subdir):
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


def get_device():
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


device = get_device()


def load_model(model, exclude_modules, freeze_modules):
    load_path = "models/dilated_CNN/pretrained_weights.pt"
    if load_path:
        checkpoint = torch.load(load_path, map_location=device)
        checkpoint_dict = checkpoint["model_state_dict"]
        model_state_dict = model.state_dict()

        # filter out unnecessary keys
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


def run():
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

    model = CausalCNNEncoderOld(**network_params)
    model = model.to(device)

    # Load pretrained model weights
    # TODO: Somehow doesn't properly work yet.
    freeze_modules = ["network\.0\.network\.[01234].*"]
    print("Freezing modules:", freeze_modules)
    exclude_models = None

    model = load_model(model, exclude_models, freeze_modules)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # pos_weights = ((calc_pos_weights(loader) - 1) * .5) + 1
    # TODO: Implement this
    pos_weights = None

    loss_fn = BinaryFocalLoss(
        device=device,
        gamma=2,
        pos_weight=pos_weights,
    )

    num_epochs = 1

    train_metrics_manager = MetricsManager(
        num_epochs=num_epochs, num_classes=24, num_batches=len(train_loader)
    )

    validation_metrics_manager = MetricsManager(
        num_epochs=num_epochs, num_classes=24, num_batches=len(validation_loader)
    )

    for epoch in range(num_epochs):
        # Train
        model.train()

        with tqdm(
            train_loader,
            desc=f"Training dilated CNN. Epoch {epoch +1}",
            total=len(train_loader),
        ) as pbar:
            for batch_i, (filenames, waveforms, labels) in enumerate(pbar):
                optimizer.zero_grad()

                waveforms = waveforms.float().to(device)
                labels = labels.float().to(device)
                # Note: The data is read as a float64 (double precision) array but the model expects float32 (single precision).
                # So we have to convert it using .float()
                predictions_logits = model(waveforms)

                # TODO: Use proper loss
                # TODO: Move this into the model
                predictions_probabilities = torch.sigmoid(predictions_logits)
                predictions = torch.round(predictions_probabilities)
                loss = loss_fn(predictions, labels, model.training)

                save_prediction(filenames, predictions, predictions_probabilities)
                save_label(filenames, "data/cinc2020_flattened", "output", "training")

                # Report loss and accuracy (or whatever metric is important) during training for the training set

                # After each epoch, report more metrics for the validation set. Maybe not all of them.

                # At teh env, report everything for the test set

                loss.backward()
                optimizer.step()

                train_metrics_manager.update_confusion_matrix(
                    labels, predictions, epoch
                )
                train_metrics_manager.update_loss(loss, epoch, batch_i)

                # CUDA  print(f" > loss: {loss.item()} \n")

                # Update tqdm postfix with current loss and accuracy
                pbar.set_postfix(loss=f"\n{loss.item():.3f}")

                # Update tqdm progress
                pbar.update()

            train_metrics_manager.compute_metrics(epoch)
            train_metrics_manager.report(epoch)

        # Validation evaluation
        model.eval()

        with tqdm(
            train_loader,
            desc=f"Evaluate validation on dilated CNN. Epoch {epoch +1}",
            total=len(train_loader),
        ) as pbar:
            # Train
            for batch_i, (filenames, waveforms, labels) in enumerate(pbar):
                with torch.no_grad():
                    for batch_i, (filenames, waveforms, labels) in enumerate(
                        validation_loader
                    ):
                        waveforms = waveforms.float().to(device)
                        labels = labels.float().to(device)

                        predictions = model(waveforms)
                        predictions = torch.sigmoid(predictions)
                        predictions = torch.round(predictions)

                        loss = loss_fn(predictions, labels, model.training)
                        validation_metrics_manager.update_loss(loss, epoch, batch_i)
                        validation_metrics_manager.update_confusion_matrix(
                            labels, predictions, epoch
                        )

                        save_prediction(filenames, labels, predictions)
                        save_label(
                            filenames, "data/cinc2020_flattened", "output", "validation"
                        )

            validation_metrics_manager.compute_metrics(epoch)
            train_metrics_manager.report(epoch)

    print(
        "Run the challenges e valuation code e.g.: python utils/evaluate_12ECG_score.py output/training output/predictions"
    )

    # TODO: train
