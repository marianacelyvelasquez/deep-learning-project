import wfdb
import wfdb.processing
import numpy as np
import scipy
import scipy.io
from scipy import misc, interpolate
import torchvision.transforms as transforms
import torch


def select_segment(ecg_signal, duration, fs_target):
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


def our_process_record(record_path, their=False, fancy_select=False):
    # Read a ECG measurement (record) from the CINC2020 dataset.
    # It read the .mat and .hea file and creates a record object out of it.
    # Note: We do not have an Annotations object. Annotation objects can be used
    # to add labels to any element in the time series. The CINC2020 dataset
    # doesn't label any specific time point in the time series. Instead, it
    # labels the whole time series with a diagnosis.

    record = wfdb.rdrecord(record_path)
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
            if their is not True:
                x_tmp, _ = wfdb.processing.resample_sig(x_tmp, fs, fs_target)
            else:
                length = len(x_tmp)
                x = np.linspace(0, length / fs, num=length)
                f = interpolate.interp1d(x, x_tmp, axis=0)
                xnew = np.linspace(
                    0,
                    length / fs,
                    num=int((length / fs) * 500),
                )
                x_tmp = f(xnew)  # use interpolation function returned by `interp1d`

        # Fix to given duration if necessary
        if len(x_tmp) > N:
            # Take first {duration} seconds of resampled signal
            # x_tmp = x_tmp[:N]
            if fancy_select is True:
                x_tmp = select_segment(x_tmp, duration, fs_target)
            else:
                x_tmp = x_tmp[:N]
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

    return ecg_signal


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


def their_process_record(record_path):
    max_sample_length = 5000

    waveform = scipy.io.loadmat(record_path + ".mat")["val"]
    length = waveform.shape[1]

    with open(record_path + ".hea", "r") as f:
        header = parse_header(f.readlines())

    if header["sample_Fs"] != 500:
        print("Resampling signal to 500Hz")
        x = np.linspace(0, length / header["sample_Fs"], num=length)
        f = interpolate.interp1d(x, waveform, axis=1)
        xnew = np.linspace(
            0,
            length / header["sample_Fs"],
            num=int((length / header["sample_Fs"]) * 500),
        )
        waveform = f(xnew)  # use interpolation function returned by `interp1d`

    if max_sample_length:
        length = np.min([waveform.shape[1], max_sample_length])
        waveform_padded = np.zeros((waveform.shape[0], max_sample_length))
        waveform_padded[:, 0:length] = waveform[:, 0:length]

    sample = {
        "waveform": waveform_padded if max_sample_length else waveform,
        "header": header,
        "label": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        "length": length,
    }

    transform = transforms.Compose([ApplyGain(umc=False), ToTensor()])
    sample = transform(sample)

    return sample


if __name__ == "__main__":
    record_path = "data/cinc2020/training/ptb/g1/S0001"  # 38k duration vs 1k hz
    record_path = "data/cinc2020/training/cpsc_2018_extra/g1/Q0002"  # 30min or so
    record_original = wfdb.rdrecord(record_path)

    k = record_original.fs // 500

    print("")
    print("record_original.shape: ", record_original.p_signal.shape)
    print("record_original.fs: ", record_original.fs)
    print("")

    ecg_signal_our = our_process_record(record_path, their=False, fancy_select=True)

    print("")
    print("ecg_signal_our.shape: ", ecg_signal_our.shape)
    print("")

    ecg_signal_their = their_process_record(record_path)

    print("")
    print("ecg_signal_our.shape: ", ecg_signal_our.shape)
    print("")

    # Plot
    import matplotlib.pyplot as plt

    # Plotting the ECG signals

    # Creating a figure with 12 subplots, one for each channel
    fig, axes = plt.subplots(2, 1, figsize=(15, 20))

    # Plotting each channel
    start_time = 0
    stop_time = 1000
    for i in range(1):
        # Original signal
        axes[i].plot(
            np.arange(start_time, start_time + stop_time * k),
            record_original.p_signal[
                start_time * k : start_time * k + stop_time * k, i
            ],
            label="Original",
            color="blue",
            marker=".",
        )

        # Our processed signal
        # Plotting every 2nd point as it's downsampled to 500Hz
        axes[i].plot(
            np.arange(start_time, start_time + stop_time * k, k),
            ecg_signal_our[start_time : start_time + stop_time, i],
            label="Our Processed",
            color="green",
            linestyle="--",
            marker=".",
        )

        # Their processed signal
        axes[i].plot(
            np.arange(start_time, start_time + stop_time * k, k),
            ecg_signal_their["waveform"][i, start_time : start_time + stop_time],
            label="Their Processed",
            color="red",
            linestyle=":",
            marker=".",
        )

        axes[i].set_title(f"Channel {i+1}")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig("compare_signal_processing.png")
    plt.show()
