import os
import shutil
from pathlib import Path
from tqdm import tqdm
from dataloaders.cinc2020.dataset import Cinc2020Dataset
import torch

from torch.utils.data import DataLoader, random_split


def preprocess_dataset():
    dataset = Cinc2020Dataset()

    # Fix randomness for reproducibility
    generator = torch.Generator().manual_seed(42)

    test_length = int(len(dataset) * 0.2)
    validation_length = int(len(dataset) * 0.1)
    train_length = len(dataset) - test_length - validation_length

    # Test, train, validation split
    train_data, test_data, validation_data = random_split(
        dataset, [train_length, test_length, validation_length], generator=generator
    )

    # train_data has shape: Array('stringofrecordname', 2D array of floats, 1D array (size 24) of 0s and one 1)


    # MARI
    if isinstance(dataset, torch.utils.data.Dataset):
        print(f"dataset is a PyTorch dataset with train_data[0]: {train_data[0]} , train_data[0] length: {len(train_data[0])}") 
    else:
        print("dataset is not a PyTorch dataset")
    # not MARI anymore

    # train_freq = self.get_label_frequencies(train_loader) / len(train_loader)
    # test_freq = self.get_label_frequencies(test_loader) / len(test_loader)
    # validation_freq = self.get_label_frequencies(validation_loader) / len(
    #     validation_loader
    # )

    # Labels for each list
    labels = ["Train freq.", "Test freq. ", "Valid freq."]

    # Function to print the lists as a table with labels
    print("\nLabel frequencies:")
    # for label, lst in zip(labels, [train_freq, test_freq, validation_freq]):
    #     row = f"{label}: " + " ".join(f"{val:6.2f}" for val in lst)
    #     print(row)
    # print("\n")

    return (
        train_data,
        validation_data,
        test_data,
    )


# This is the part of the script that handles command line arguments
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "lets-go":
        print("Entering pre-processing mode")
        preprocess_dataset()
    else:
        print("Usage: python preprocess_dataset.py <path_to_cinc2020_directory>")


    
