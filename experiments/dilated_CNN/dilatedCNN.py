import re
from tqdm import tqdm
import torch

from dataloaders.cinc2020.dataset import Cinc2020Dataset
from torch.utils.data import DataLoader
from models.dilated_CNN.model import CausalCNNEncoder as CausalCNNEncoderOld

network_params = {
    'in_channels': 12,
    'channels': 108,
    'depth': 6,
    'reduced_size': 216,
    'out_channels': 111,
    'kernel_size': 3
}


def load_model(model, exclude_modules, freeze_modules):
    load_path = 'models/dilated_CNN/cinc2020_7/pretraining/classes24_epoch_6.pt'

    if load_path:
        checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
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


def train():
    dataset = Cinc2020Dataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = CausalCNNEncoderOld(**network_params)

    # Load pretrained model weights
    # TODO: Somehow doesn't properly work yet.
    """
    freeze_modules = ['network\.0\.network\.[01234].*']
    print('Freezing modules:', freeze_modules)
    exclude_models = None

    model = load_model(model, exclude_models, freeze_modules)
    """

    model.train()

    for idx, (waveforms, labels) in enumerate(tqdm(loader, desc="Training dialted CNN", total=len(loader))):
        # Note: The data is read as a float64 (double precision) array but the model expects float32 (single precision).
        # So we have to convert it using .float()
        output = model(waveforms.float())
