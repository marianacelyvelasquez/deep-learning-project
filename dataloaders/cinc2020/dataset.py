import os
import wfdb
import numpy as np
import pandas as pd
import wfdb.processing
from torch.utils.data import Dataset
from pathlib import Path

from experiments.dilated_CNN.config import Config


class Cinc2020Dataset(Dataset):
    def __init__(self, records):
        # self.root_dir = "data/cinc2020/training"
        self.root_dir = Config.DATA_DIR
        self.records = records

        # Equivalence classes
        self.eq_classes = np.array(
            [
                ["713427006", "59118001"],
                ["284470004", "63593006"],
                ["427172004", "17338001"],
            ]
        )

        # Source: https://github.com/physionetchallenges/physionetchallenges.github.io/blob/master/2020/Dx_map.csv
        mappings = pd.read_csv("data/cinc2020/label_cinc2020_top24.csv", delimiter=",")
        self.labels_map = mappings["SNOMED CT Code"].values

        # NOTE: In each g* directory, there is a file RECORDS.
        # This file contains all the records in the given g* subdirectory
        # but there is an error: The first g* dir actually only contains
        # 999 and not 1000 entries. I wrote them about it.
        # Conclusion: Shouldn't matter for us.

        # TODO: Somehow fix the ordering of the samples for reproducibility.

        # Loop over contents of all subdirectories
        """
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Read all records from a current g* subdirectory.
            for filename in filenames:
                if filename.endswith(".hea"):
                    record_path = os.path.join(dirpath, filename.split(".")[0])
                    self.records.append(record_path)
        """

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        # TODO: Is idx always an int? Does PyTorch handle returning whole batches for us?

        # Read a ECG measurement (record) from the CINC2020 dataset.
        # It read the .mat and .hea file and creates a record object out of it.
        # Note: We do not have an Annotations object. Annotation objects can be used
        # to add labels to any element in the time series. The CINC2020 dataset
        # doesn't label any specific time point in the time series. Instead, it
        # labels the whole time series with a diagnosis.
        record = wfdb.rdrecord(self.records[idx])
        ecg_signal = record.p_signal

        # Fix parameters for our target timeseries
        fs = record.fs  # Original sampling frequency
        fs_target = 500  # Target sampling frequency
        duration = 10  # seconds
        N = duration * fs_target  # Number of time points in target timeseries
        lx = np.zeros((ecg_signal.shape[1], N))  # Allocate memory

        # We loop over all 12 leads/channels and resample them to the target frequency.
        # WFDB has a function for that but assumes there's an annotation object,
        # which we don't have. So we have to do it manually.
        for chan in range(ecg_signal.shape[1]):
            # Choose lead
            x_tmp = ecg_signal[:, chan]

            # Resample to target frequency if necessary
            if fs != fs_target:
                x_tmp, _ = wfdb.processing.resample_sig(
                    ecg_signal[:, chan], fs, fs_target
                )

            # Fix to given duration if necessary
            if len(x_tmp) > N:
                # Take first {duration} seconds of resampled signal
                x_tmp = x_tmp[:N]
            elif len(x_tmp) < N:
                # Right pad with zeros to given duration
                # It's important we append the zeros because
                # our data has a "temporal direction".
                x_tmp = np.pad(x_tmp, (0, N - len(x_tmp)))

            # Store in lx
            lx[chan] = x_tmp

        # TODO: We should probably normalize the signal to zero mean and unit variance.
        # I think we do that in the dataloader though.
        ecg_signal = lx

        assert ecg_signal.shape == (12, 5000), "A signal has wrong shape."

        # TODO: Currently assuming that the 3 field in comments is the diagnosis.
        # This might or might not be always true.
        diagnosis_string = record.comments[2].split(": ", 1)[1].strip()
        diagnosis_list = diagnosis_string.split(",")

        # Replace diagnosis with equivalent class if necessary
        for diagnosis in diagnosis_list:
            if diagnosis in self.eq_classes[:, 1]:
                # Get equivalent class
                eq_class = self.eq_classes[self.eq_classes[:, 1] == diagnosis][0][0]
                # Replace diagnosis with equivalent class
                diagnosis_list = [
                    eq_class if x == diagnosis else x for x in diagnosis_list
                ]

        diagnosis_list = [int(diagnosis) for diagnosis in diagnosis_list]

        # Binary encode labels. 1 if label is present, 0 if not.
        labels_binary_encoded = np.isin(self.labels_map, diagnosis_list).astype(int)

        assert len(labels_binary_encoded) == 24, "Wrong number of labels."

        # We return the filename because it's an indicator for the current record.
        # We need it to be able to store the perdictions using the same filename.
        # The challenge requires us to do this.
        filename = Path(self.records[idx]).stem

        return filename, ecg_signal, labels_binary_encoded

        """
        # This is just needed for plotting:
        # We need to create a record from the resampled signal so we can plot it.
        record_resampled = wfdb.Record(
            record_name='A0001',
            n_sig=lx.shape[1],
            fs=fs_target,
            sig_len=lx.shape[0],
            file_name='A0001',
            fmt='212',
            adc_gain=np.ones(lx.shape[1]),
            baseline=np.zeros(lx.shape[1]),
            units=np.array(['mV'] * lx.shape[1]),
            sig_name=np.array(['ECG'] * lx.shape[1]),
            p_signal=lx
        )

        wfdb.plot_wfdb(record=record)
        wfdb.plot_wfdb(record=record_resampled)
        """
