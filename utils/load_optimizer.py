import torch
import torch.optim as optim

def load_optimizer(model, device, checkpoint_path, load_optimizer: bool, learning_rate=0.001):
    print(f"Loading optimizer with learning rate {learning_rate} and checkpoint_path {checkpoint_path}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if load_optimizer is False:
        print("Not loading optimizer state dict because checkpoint_path is None")

        return optimizer

    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"Loading optimizer state dict from {checkpoint_path}")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return optimizer
