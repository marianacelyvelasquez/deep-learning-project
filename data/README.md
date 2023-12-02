# Introduction 

There are two datasets. One from te 2020 challenge and one form the 2021 challenge. ECG data is usually given in the [WFDB format](https://physionet.org/physiotools/wpg/wpg.pdf).

## 2020 Data
- You can find the data and additional information on https://physionet.org/content/challenge-2020/1.0.2/
- https://physionet.org/lightwave/?db=challenge-2020/1.0.2

The data can by downloaded by running (~4.5 GB)

`wget -r -N -c -np -P cinc2020 https://physionet.org/files/challenge-2020/1.0.2/`

or via http from the source above.

The downloaded data is split into folders, each folder represents one of the data sources. Furthermore in each dataset folder the files are grouped into subfolders with up to 1000 records per subfolder. These subfolders are named as g# where the # starts at 1. Once 1000 records are allocated to a folder a new folder is started with the # incremented by one.

## 2021 Data
- You can find the data and additional information on https://physionet.org/content/challenge-2021/1.0.3/
- A data visulaization tool can be found at https://physionet.org/lightwave/?db=challenge-2021/1.0.3

The data can be downloaded by running

`wget -r -N -c -np -P 2021 https://physionet.org/files/challenge-2021/1.0.3/`

or via http from the source above.

The downloaded data is split into folders, each folder represents one of the data sources.