from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
import wfdb

class Processor:
    def __init__(self, input_dir):
        # input_dir could be "data/cinc2020_flattened" or "data/cinc2020_tiny"
        self.input_dir = input_dir
        self.eq_classes = np.array([
            ["713427006", "59118001"],
            ["284470004", "63593006"],
            ["427172004", "17338001"],
        ])

        # Source: https://github.com/physionetchallenges/physionetchallenges.github.io/blob/master/2020/Dx_map.csv
        mappings = pd.read_csv("data/cinc2020/label_cinc2020_top24.csv", delimiter=",")
        self.labels_map = mappings["SNOMED CT Code"].values

        # NOTE: In each g* directory, there is a file RECORDS.
        # This file contains all the records in the given g* subdirectory
        # but there is an error: The first g* dir actually only contains
        # 999 and not 1000 entries. I wrote them about it.
        # Conclusion: Shouldn't matter for us.

        # TODO: Somehow fix the ordering of the samples for reproducibility.


    def process_records(self, record_paths, output_dir):

        # Read a ECG measurement (record) from the CINC2020 dataset.
        # It read the .mat and .hea file and creates a record object out of it.
        # Note: We do not have an Annotations object. Annotation objects can be used
        # to add labels to any element in the time series. The CINC2020 dataset
        # doesn't label any specific time point in the time series. Instead, it
        # labels the whole time series with a diagnosis.
        for record_path in record_paths:
            record = wfdb.rdrecord(record_path)
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
                        x_tmp, _ = wfdb.processing.resample_sig(x_tmp, fs, fs_target)

                # Fix to given duration if necessary
                if len(x_tmp) > N:
                        # Take first {duration} seconds of resampled signal
                        x_tmp = x_tmp[:N]
                elif len(x_tmp) < N:
                        # Right pad with zeros to given duration
                        # It's important we append the zeros because
                        # our data has a "temporal direction".
                        x_tmp = np.pad(x_tmp, (0, N - len(x_tmp)))
                lx[chan] = x_tmp

            # TODO: We should probably normalize the signal to zero mean and unit variance.
            # I think we do that in the dataloader though.
            ecg_signal = lx

            # MARI RIC: Rest of the code for processing labels remains unchanged

            # MARI RIC: Save the processed data as new .hea and .mat files
            new_filename = Path(output_dir) / f"{Path(record_path).stem}" # is something like "cinc2020_processed_training/g00001"
            self.save_processed_data(new_filename, ecg_signal)

    def save_processed_data(self, filename, ecg_signal):

        # Save resampled data as .hea file
        with open(f"{filename}.hea", 'w') as hea_file:
            hea_file.write(f"signal 0 {filename}.mat 1000 {len(ecg_signal)} 16 0\n")

        # Save resampled data as .mat file
        scipy.io.savemat(f"{filename}.mat", {'ECG': ecg_signal})

    # def save_processed_data(self, filename, ecg_signal):
    #     print(f"i am inside save_processed_data")

    #     # Get the original record for header information
    #     original_record = wfdb.rdrecord(f"{self.input_dir}/{Path(filename).stem}")

    #     # Save resampled data as .hea file AND .mat FILE
    #     wfdb.io.convert.matlab.wfdb_to_mat(filename, sampfrom=0, sampto=len(ecg_signal), channels=None)


