import re
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
    'in_channels': 12,
    'channels': 108,
    'depth': 6,
    'reduced_size': 216,
    'out_channels': 24,
    'kernel_size': 3
}


def get_device():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    # Check if MPS is available
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using METAL GPU')
    # If neither CUDA nor MPS is available, use CPU
    else:
        device = torch.device('cpu')
        print('Using CPU')

    return device


device = get_device()


def load_model(model, exclude_modules, freeze_modules):
    load_path = "models/dilated_CNN/pretrained_weights.pt"
    if load_path:
        checkpoint = torch.load(load_path, map_location=device)
        checkpoint_dict = checkpoint['model_state_dict']
        model_state_dict = model.state_dict()

        # filter out unnecessary keys
        if exclude_modules:
            checkpoint_dict = {
                k: v for k, v in checkpoint_dict.items()
                if k in model_state_dict and
                not any(re.compile(p).match(k) for p in exclude_modules)
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
        dataset,
        [train_length, test_length, validation_length],
        generator=generator
    )

    train_loader = DataLoader(train_data, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    validation_loader = DataLoader(
        validation_data, batch_size=128, shuffle=False)

    model = CausalCNNEncoderOld(**network_params)
    model = model.to(device)

    # Load pretrained model weights
    # TODO: Somehow doesn't properly work yet.
    freeze_modules = ['network\.0\.network\.[01234].*']
    print('Freezing modules:', freeze_modules)
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

    model.train()

    metrics_manager = MetricsManager(
        num_epochs=10, num_classes=24, num_batches=len(train_loader))

    num_epochs = 10
    for epoch in range(num_epochs):
        with tqdm(train_loader, desc=f"Training dilated CNN. Epoch {epoch +1}", total=len(train_loader)) as pbar:
            for batch_i, (waveforms, labels) in enumerate(pbar):
                optimizer.zero_grad()

                waveforms = waveforms.float().to(device)
                labels = labels.float().to(device)
                # Note: The data is read as a float64 (double precision) array but the model expects float32 (single precision).
                # So we have to convert it using .float()
                predictions = model(waveforms)

                # TODO: Use proper loss
                # TODO: Move this into the model
                predictions = torch.sigmoid(predictions)
                predictions = torch.round(predictions)
                loss = loss_fn(predictions, labels, model.training)

                # Report loss and accuracy (or whatever metric is important) during training for the training set

                # After each epoch, report more metrics for the validation set. Maybe not all of them.

                # At teh env, report everything for the test set

                loss.backward()
                optimizer.step()

                metrics_manager.update(
                    labels, predictions, loss, epoch, batch_i)

                # CUDA  print(f" > loss: {loss.item()} \n")

                # Update tqdm postfix with current loss and accuracy
                pbar.set_postfix(loss=f"\n{loss.item():.3f}")

                # Update tqdm progress
                pbar.update()

    # TODO: train
