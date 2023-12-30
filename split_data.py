#!/usr/bin/env python3
import os
import sys
import random


def create_symlinks(source_dir, dest_dir, percentage=5):
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Extracting base filenames without extensions
    base_files = set(
        os.path.splitext(f)[0]
        for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f))
        and not f.startswith(".")
        and f.endswith(".mat")
    )

    num_files_to_link = max(1, len(base_files) * percentage // 100)

    if num_files_to_link == 0:
        print("No non-hidden files to link.")
        return

    selected_bases = random.sample(base_files, num_files_to_link)

    for base in selected_bases:
        for ext in [".hea", ".mat"]:
            file = f"{base}{ext}"
            src_path = os.path.abspath(
                os.path.join(source_dir, file)
            )  # Convert to absolute path
            dest_path = os.path.join(dest_dir, file)
            if os.path.exists(src_path):
                os.symlink(src_path, dest_path)
                print(f"Created symlink with absolute path for {file} in {dest_dir}")
            else:
                print(f"File {file} does not exist in the source directory.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 script.py <source_directory> <destination_directory>")
        sys.exit(1)

    source_directory = sys.argv[1]
    destination_directory = sys.argv[2]

    create_symlinks(source_directory, destination_directory)
