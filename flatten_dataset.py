import os
import shutil
from pathlib import Path
from tqdm import tqdm


def flatten_dataset(source_dir, flattened_dir):
    if not os.path.exists(flattened_dir):
        os.makedirs(flattened_dir)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # List of files to move
    files_to_move = []

    # Walk through the source directory and list all .hea and .mat files
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename.endswith(".hea") or filename.endswith(".mat"):
                files_to_move.append(
                    (
                        os.path.join(dirpath, filename),
                        os.path.join(flattened_dir, filename),
                    )
                )

    # Move the files with a progress bar
    for src_file, dst_file in tqdm(
        files_to_move, desc="Flattening dataset", unit="file"
    ):
        src_file = os.path.join(current_dir, str(src_file))
        dst_file = os.path.join(current_dir, (dst_file))

        if not os.path.exists(dst_file):  # Check if file already exists
            os.symlink(str(src_file), str(dst_file))
        else:
            print(f"File {dst_file} already exists. Skipping.")

    print(f"Flattened dataset is available at: {flattened_dir}")


# This is the part of the script that handles command line arguments
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python flatten_dataset.py <path_to_cinc2020_directory>")
    else:
        flatten_dataset(sys.argv[1], sys.argv[2])
