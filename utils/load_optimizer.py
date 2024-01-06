import torch
from experiments.dilated_CNN.dilatedCNN import dilatedCNNExperiment
import torch.optim as optim

def load_optimizer(model, device, checkpoint_path, load_optimizer: bool, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    if load_optimizer is False and checkpoint_path is None:
        return optimizer

    if load_optimizer is False and checkpoint_path is True:
        print("Not loading optimizer state dict because LOAD_OPTIMIZER is False")
        return optimizer

    if load_optimizer is True and checkpoint_path is None:
        print("Not loading optimizer state dict because checkpoint_path is None")
        return optimizer

    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"Loading optimizer state dict from {checkpoint_path}")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return optimizer
