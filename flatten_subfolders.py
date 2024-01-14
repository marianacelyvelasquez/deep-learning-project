import os
import shutil
from pathlib import Path
from tqdm import tqdm
from flatten_dataset import flatten_dataset

def flatten_subfolders(parent_folder):
    parent_folder_path = Path(parent_folder)
    flattened_parent_folder = f"{parent_folder_path}_flattened_by_source"

    # Iterate over subfolders in parent folder
    for subfolder in parent_folder_path.iterdir():
        if subfolder.is_dir():
            flattened_subfolder = Path(flattened_parent_folder) / (subfolder.name + "_flattened")
            flatten_dataset(subfolder, flattened_subfolder)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python flatten_subfolders.py <path_to_parent_folder>")
    else:
        flatten_subfolders(sys.argv[1])
