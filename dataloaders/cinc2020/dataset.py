import os
import wfdb
import numpy as np
from torch.utils.data import Dataset


class Cinc2020Dataset(Dataset):
    def __init__(self):
        self.root_dir = 'data/cinc2020/training'

        self.records = []

        # NOTE: In each g* directory, there is a file RECORDS.
        # This file contains all the records in the given g* subdirectory
        # but there is an error: The first g* dir actually only contains
        # 999 and not 1000 entries. I wrote them about it.
        # Conclusion: Shouldn't matter for us.

        # TODO: Somehow fix the ordering of the samples for reproducibility.

        # Loop over contents of all subdirectories
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Read all records from a current g* subdirectory.
            for filename in filenames:
                if filename == 'RECORDS':
                    records_file = os.path.join(dirpath, filename)

                    with open(records_file, 'r') as file:
                        # Create list of paths to records
                        records_names = file.read().splitlines()
                        records_paths = [os.path.join(
                            dirpath, record_name) for record_name in records_names]
                        self.records.extend(records_paths)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # TODO: Handle different sampling sizes.
            # wfdb has some limited support for that.
            record = wfdb.rdrecord(self.records[idx])
            ecg_signal = record.p_signal

            # TODO: Currently assuming that the 3 field in comments is the diagnosis.
            # This might or might not be always true.
            diagnosis_string = record.comments[2].split(': ', 1)[1].strip()
            diagnosis_list = diagnosis_string.split(',')

            return ecg_signal, diagnosis_list
