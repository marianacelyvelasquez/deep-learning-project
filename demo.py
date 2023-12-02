from dataloaders.cinc2020.dataset import Cinc2020Dataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    dataset = Cinc2020Dataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Doesnt work yet since we have different sampling sizes
    # train_features, train_labels = next(iter(loader))
