import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
import wfdb
import wfdb.processing
import wfdb.io.convert


class Processor:
    def __init__(self, input_dir):
        # input_dir could be "data/cinc2020_flattened" or "data/cinc2020_tiny"
        self.input_dir = input_dir
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

    def select_segment(self, ecg_signal, duration, fs_target):
        N = duration * fs_target
        start_idx = 0
        end_idx = N

        while end_idx <= len(ecg_signal):
            segment = ecg_signal[start_idx:end_idx]

            # Check for zeros and move the window if necessary
            if np.any(segment == 0):
                start_idx += fs_target  # Move window by 1 second
                end_idx += fs_target
                continue

            # Check for constant values in the segment
            if len(np.unique(segment)) < (0.1 * N):  # Threshold for variety
                start_idx += fs_target  # Move window by 1 second
                end_idx += fs_target
                continue

            return segment

        # If no suitable segment is found, return the initial N elements
        return ecg_signal[:N]

    def process_records(self, record_paths, output_dir):
        # Read a ECG measurement (record) from the CINC2020 dataset.
        # It read the .mat and .hea file and creates a record object out of it.
        # Note: We do not have an Annotations object. Annotation objects can be used
        # to add labels to any element in the time series. The CINC2020 dataset
        # doesn't label any specific time point in the time series. Instead, it
        # labels the whole time series with a diagnosis.

        for record_path in record_paths:
            record = wfdb.rdrecord(record_path)
            print("record_path: ", record_path)
            ecg_signal = record.p_signal
            # print(f"first record.fmt {record.fmt} \n\n")
            # print("Number of channels:", record.n_sig)

            # Fix parameters for our target timeseries
            fs = record.fs  # Original sampling frequency
            fs_target = 500  # Target sampling frequency
            duration = 10  # seconds
            N = duration * fs_target  # Number of time points in target timeseries
            lx = np.zeros((N, ecg_signal.shape[1]))  # Allocate memory
            # print(f"ecg_signal shape: {ecg_signal.shape} \n\n")

            # We loop over all 12 leads/channels and resample them to the target frequency.
            # WFDB has a function for that but assumes there's an annotation object,
            # which we don't have. So we have to do it manually.
            for chan in range(ecg_signal.shape[1]):
                # Choose lead
                x_tmp = ecg_signal[:, chan]

                # TODO: Make this better.
                # We replace nan with 0.0.
                x_tmp = np.nan_to_num(x_tmp)

                # Resample to target frequency if necessary
                if fs != fs_target:
                    x_tmp, _ = wfdb.processing.resample_sig(x_tmp, fs, fs_target)

                # Fix to given duration if necessary
                if len(x_tmp) > N:
                    # Take first {duration} seconds of resampled signal
                    # x_tmp = x_tmp[:N]
                    x_tmp = self.select_segment(x_tmp, duration, fs_target)
                elif len(x_tmp) < N:
                    # Right pad with zeros to given duration
                    # It's important we append the zeros because
                    # our data has a "temporal direction".
                    x_tmp = np.pad(x_tmp, (0, N - len(x_tmp)))
                x_tmp = np.resize(x_tmp, (N,))
                lx[:, chan] = x_tmp

            # TODO: We should probably normalize the signal to zero mean and unit variance.
            # I think we do that in the dataloader though.
            ecg_signal = lx
            # print(f"ecg_signal shape after resampling: {ecg_signal.shape} \n\n")
            # MARI: new record
            # Create new record object with the resampled signal
            new_record = wfdb.Record(
                p_signal=ecg_signal,
                record_name=record.record_name,
                fs=fs_target,
                sig_name=record.sig_name,
                units=record.units,
                comments=record.comments,
                base_date=record.base_date,
                base_time=record.base_time,
                fmt=record.fmt,
            )

            # MARI RIC: Rest of the code for processing labels remains unchanged

            # MARI RIC: Save the processed data as new .hea and .mat files
            new_filename = (
                Path(output_dir) / f"{Path(record_path).stem}"
            )  # is something like "cinc2020_processed_training/g00001"
            self.save_processed_data(new_filename, new_record)

    def save_processed_data(self, filename, record):
        string_filename = str(filename)
        # print(f"\nfilename: {filename} \n")
        # Save record as .hea and .dat files
        # print(f"fmt issue? : {record.fmt} \n")
        # PROBLEMS HERE: (is my ecg_signal wrongly shaped ?)

        # for ch in range(np.shape(self.p_signal)[1]):
        #     # Get the minimum and maximum (valid) storage values
        #     dmin, dmax = _digi_bounds(self.fmt[ch])
        #     # add 1 because the lowest value is used to store nans
        #     dmin = dmin + 1

        wfdb.io.wrsamp(
            os.path.basename(string_filename),
            record.fs,
            record.units,
            record.sig_name,
            record.p_signal,
            comments=record.comments,
            fmt=record.fmt,
            write_dir=os.path.dirname(string_filename),
        )

        # Save resampled data as .hea file AND .mat FILE ???
        wfdb.io.convert.matlab.wfdb_to_mat(string_filename)

        # Move the file from root directory to string_filename directory
        data_dir = os.path.dirname(string_filename)
        filename = os.path.basename(string_filename)

        mat_file_source = filename + "m.mat"
        mat_file_target = os.path.join(data_dir, filename + "m.mat")
        shutil.move(mat_file_source, mat_file_target)

        hea_file_source = filename + "m.hea"
        hea_file_original = os.path.join(data_dir, filename + ".hea")
        hea_file_target = os.path.join(data_dir, filename + "m.hea")
        shutil.move(hea_file_source, hea_file_target)
        os.remove(hea_file_original)

        dat_file = os.path.join(data_dir, filename + ".dat")
        os.remove(dat_file)
