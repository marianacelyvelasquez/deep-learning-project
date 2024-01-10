import torch
import re
from models.dilated_CNN.model import CausalCNNEncoder as CausalCNNEncoderOld


def get_map_location(device):
    if device.type == "cuda":
        return f"cuda:{device.index}"

    return device.type


def load_model(
    network_params,
    device,
    checkpoint_path,
):
    model = CausalCNNEncoderOld(**network_params)
    model = model.to(device)

    # Load pretrained model weights
    # TODO: Somehow doesn't properly work yet.
    freeze_modules = ["network\.0\.network\.[01234].*"]
    exclude_modules = None

    print(f"Loading pretrained dilated CNN from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=get_map_location(device))
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

    print("Freezing modules:", freeze_modules)
    for k, param in model.named_parameters():
        if any(re.compile(p).match(k) for p in freeze_modules):
            param.requires_grad = False

    return model
