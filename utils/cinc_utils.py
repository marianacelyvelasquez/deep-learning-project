import os
import torch
import numpy as np
import pandas as pd
import fnmatch
import scipy.io
from skmultilearn.model_selection.iterative_stratification import (
    iterative_train_test_split,
)


import torch
import numpy as np


class ToTensorPredict(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, waveform):
        waveform = torch.from_numpy(waveform).type(torch.FloatTensor)
        return waveform


class ApplyGainPredict(object):
    """Normalize ECG signal by multiplying by specified gain and converting to millivolts."""

    def __call__(self, waveform):
        # CINC data is only multiplied with 0.001, not 4.88
        waveform = waveform * 0.001
        return waveform


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        waveform = sample["waveform"]
        sample["waveform"] = torch.from_numpy(waveform).type(torch.FloatTensor)
        sample["label"] = torch.from_numpy(np.array(sample["label"])).type(
            torch.FloatTensor
        )
        return sample


class ApplyGain(object):
    """Normalize ECG signal by multiplying by specified gain and converting to millivolts."""

    def __init__(self, umc=False):
        self.umc = umc

    def __call__(self, sample):
        if self.umc:
            waveform = sample["waveform"] * 0.001 * 4.88
        else:
            waveform = sample["waveform"] * 0.001
        sample["waveform"] = waveform
        return sample


def get_classes(root_dir, header_files):
    classes = set()
    for file in header_files:
        input_file = os.path.join(root_dir, file)
        with open(input_file, "r") as f:
            for lines in f:
                if lines.startswith("#Dx"):
                    tmp = lines.split(": ")[1].split(",")
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)


class To12Lead(object):
    """Convert 8 lead waveforms to their 12 lead equivalent."""

    def __call__(self, sample):
        waveform = sample["waveform"]

        out = np.zeros((12, waveform.shape[1]))
        out[0:2, :] = waveform[0:2, :]  # I and II
        out[2, :] = waveform[1, :] - waveform[0, :]  # III = II - I
        out[3, :] = -(waveform[0, :] + waveform[1, :]) / 2  # aVR = -(I + II)/2
        out[4, :] = waveform[0, :] - (waveform[1, :] / 2)  # aVL = I - II/2
        out[5, :] = waveform[1, :] - (waveform[0, :] / 2)  # aVF = II - I/2
        out[6:12, :] = waveform[2:8, :]  # V1 to V6

        sample["waveform"] = out
        return sample


def generate_path(root_dir, filename):
    return os.path.join(root_dir, filename)


def parse_header(header_data):
    tmp_hea = header_data[0].split(" ")
    ptID = tmp_hea[0]
    num_leads = int(tmp_hea[1])
    sample_Fs = int(tmp_hea[2])
    gain_lead = np.zeros(num_leads)

    for ii in range(num_leads):
        tmp_hea = header_data[ii + 1].split(" ")
        gain_lead[ii] = int(tmp_hea[2].split(".")[0])

    # for testing, we included the mean age of 57 if the age is a NaN
    # This value will change as more data is being released
    for iline in header_data:
        if iline.startswith("# Age"):
            tmp_age = iline.split(": ")[1].strip()
            age = int(tmp_age if tmp_age != "NaN" else 57)
        elif iline.startswith("# Sex"):
            tmp_sex = iline.split(": ")[1]
            sex = 1 if tmp_sex.strip() == "Female" else 0
        elif iline.startswith("# Dx"):
            labels = [l.strip() for l in iline.split(": ")[1].split(",")]
    features = {
        "ptID": ptID,
        "labels": labels,
        "age": age,
        "sex": sex,
        "num_leads": num_leads,
        "sample_Fs": sample_Fs,
        "gain": gain_lead,
    }
    return features


def get_y(root_dir, classes, header_files, class_mapping):
    y = []
    for fname in header_files:
        with open(generate_path(root_dir, fname), "r") as f:
            header = parse_header(f.readlines())
            labels = np.zeros(len(classes))
            for l in header["labels"]:
                if class_mapping is not None:
                    l_eq = class_mapping.loc[
                        class_mapping["SNOMED CT Code"] == int(l), "Training Code"
                    ]
                    if not l_eq.empty:
                        labels[classes.index(int(l_eq))] = 1
                else:
                    labels[classes.index(int(l))] = 1
            y.append(labels)
    return np.array(y)


def get_xy(root_dir, max_sample_length, cut_off, class_mapping):
    waveform_files = sorted(fnmatch.filter(os.listdir(root_dir), "*.mat"))
    header_files = sorted(fnmatch.filter(os.listdir(root_dir), "*.hea"))
    if class_mapping is not None:
        classes = pd.unique(class_mapping["Training Code"]).tolist()
    else:
        classes = get_classes(root_dir, header_files)
    X = [f.split(".")[0] for f in waveform_files]
    if cut_off:
        keep_indexes = range(len(X))
    else:
        # filter out samples with length greather than `max_length`
        keep_indexes = []
        for idx in range(len(X)):
            waveform = scipy.io.loadmat(generate_path(root_dir, X[idx]))["val"]
            length = waveform.shape[1]
            if length <= max_sample_length:
                keep_indexes.append(idx)
    X = np.array(X)[keep_indexes]
    X = np.expand_dims(X, axis=1)
    y = get_y(root_dir, classes, header_files, class_mapping)[keep_indexes]
    return X, y, classes


def split_dataset(root_dir, test_size, max_sample_length, cut_off, class_mapping=None):
    X, y, classes = get_xy(root_dir, max_sample_length, cut_off, class_mapping)
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size)
    return X_train, y_train, X_test, y_test, classes
