import os
import shutil
from pathlib import Path
from tqdm import tqdm


def flatten_dataset(source_dir, flattened_dir_name="cinc2020_flattened"):
    # Create flattened directory if it does not exist
    flattened_dir = Path(source_dir).parent / flattened_dir_name
    flattened_dir.mkdir(exist_ok=True)

    # List of files to move
    files_to_move = []

    # Walk through the source directory and list all .hea and .mat files
    for dirpath, dirnames, filenames in os.walk(source_dir):
        # Check if this directory is a g* directory
        if Path(dirpath).name.startswith("g"):
            # Add each .hea and .mat file to the list
            for filename in filenames:
                if filename.endswith(".hea") or filename.endswith(".mat"):
                    files_to_move.append(
                        (Path(dirpath) / filename, flattened_dir / filename)
                    )

    # Move the files with a progress bar
    for src_file, dst_file in tqdm(
        files_to_move, desc="Flattening dataset", unit="file"
    ):
        shutil.move(str(src_file), str(dst_file))

    print(f"Flattened dataset is available at: {flattened_dir}")


# This is the part of the script that handles command line arguments
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python flatten_dataset.py <path_to_cinc2020_directory>")
    else:
        flatten_dataset(sys.argv[1])
