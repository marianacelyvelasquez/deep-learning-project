from dataloaders.cinc2020.dataset import Cinc2020Dataset
from torch.utils.data import DataLoader


def custom_collate(batch):
    breakpoint()


if __name__ == '__main__':
    dataset = Cinc2020Dataset()
    loader = DataLoader(dataset, batch_size=64,
                        shuffle=False)

    for batch_idx, (train_features, train_labels) in enumerate(loader):
        print(f"Batch {batch_idx} has shape {train_features.shape}")
