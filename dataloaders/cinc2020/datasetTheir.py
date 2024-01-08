from torch.utils.data import Dataset
from scipy import interpolate
import scipy.io
import os
import numpy as np
import torchvision.transforms as transforms

import utils.cinc_utils as cinc_utils


class Cinc2020Dataset(Dataset):
    """Computing in Cardiology 2020 challenge dataset for pretraining"""

    def __init__(
        self,
        X,
        y,
        classes,
        root_dir,
        name,
        num_leads=12,
        max_sample_length=5000,
    ):
        """
        Args:
        root_dir (string): Directory with all the datapoints.
        transform (callable, optional): Optional transform to be applied
        on a sample.
        """
        self.X = X
        self.y = y
        self.transform = self.get_transform()
        self.num_leads = num_leads
        self.max_sample_length = max_sample_length
        self.classes = classes
        self.root_dir = root_dir
        self.name = name

        print(
            "CINC2020Dataset initialized\nNumber of samples: {}\nUnique classes: {}".format(
                self.__len__(), self.classes
            )
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        waveform = scipy.io.loadmat(self.generate_path(idx, "waveform"))["val"]
        length = waveform.shape[1]

        with open(self.generate_path(idx, "header"), "r") as f:
            header = cinc_utils.parse_header(f.readlines())

        if header["sample_Fs"] != 500:
            # print("Resampling signal to 500Hz")
            x = np.linspace(0, length / header["sample_Fs"], num=length)
            f = interpolate.interp1d(x, waveform, axis=1)

            xnew = np.linspace(
                0,
                length / header["sample_Fs"],
                num=int((length / header["sample_Fs"]) * 500),
            )
            waveform = f(xnew)  # use interpolation function returned by `interp1d`

        if self.max_sample_length:
            length = np.min([waveform.shape[1], self.max_sample_length])
            waveform_padded = np.zeros((waveform.shape[0], self.max_sample_length))
            waveform_padded[:, 0:length] = waveform[:, 0:length]

        labels = self.y[idx]
        sample = {
            "waveform": waveform_padded if self.max_sample_length else waveform,
            "header": header,
            "label": labels,
            "length": length,
        }

        if self.transform:
            sample = self.transform(sample)

        # return filename, waveform, label
        return self.X[idx][0], sample["waveform"], sample["label"]

    def get_transform(self):
        transform = transforms.Compose(
            [cinc_utils.ApplyGain(umc=False), cinc_utils.ToTensor()]
        )

        return transform

    # Generate the path to the waveform or header file
    def generate_path(self, idx, type):
        ext = "mat" if type == "waveform" else "hea"
        fname = self.X[idx][0]

        return os.path.join(self.root_dir, f"{fname}.{ext}")
