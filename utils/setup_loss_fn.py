import numpy as np
import torch

from utils.loss import BinaryFocalLoss

def get_label_frequencies(loader):
    labels = []

    for _, _, label in loader:
        labels.append(label.numpy())

    labels = np.concatenate(labels, axis=0)

    freq = np.sum(labels, axis=0)

    return freq


def calc_pos_weights(loader):
    freq = get_label_frequencies(loader)
    freq = np.where(freq == 0, freq.max(), freq)

    return torch.Tensor(np.around(freq.max() / freq, decimals=1))

def setup_loss_fn(device, loader):
    # pos_weights = ((calc_pos_weights(loader) - 1) * .5) + 1
    # TODO: Make sure the values we compute make sense

    # 1. Calculate class weights based on training data
    pos_weights = calc_pos_weights(loader)

    # 2. Adjust weights: subtract 1, halve the result, then add 1
    # Subtracting 1 ensures that the most frequent class (which has a weight
    # of 1) will now have a weight of 0, and all other classes will have
    # their weights reduced by 1.  Multiplying by 0.5 reduces the difference
    # between the weights of different classes, making the weights less
    # extreme.  Adding 1 ensures that the weights are not less than 1, which
    # would give more importance to the more frequent classes, the opposite
    # of what we want when dealing with class imbalance.
    pos_weights = ((pos_weights - 1) * 0.5) + 1

    # pos_weights = ((.calc_pos_weights(.loader) - 1) * 0.5) + 1

    # 3. Set up Binary Focal Loss function with adjusted weights
    loss_fn = BinaryFocalLoss(
        device= device,
        gamma=2,
        pos_weight=pos_weights,
    )

    return loss_fn