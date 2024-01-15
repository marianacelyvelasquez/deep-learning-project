# Introduction
Group project for the Deep Learning course in HS 2023
Working on AI in ECG, in particular trying to introduce SWAG method and compare results with classical-SGD 

This repo is aiming at two main things:
1. Craft ideas and brainstorm solutions, visualize data
2. Train a model and return the results as part of the project

# Setup
NOTE: If you are on Windows, then do the below but Windows style... While I have no idea how one does it, it should be easy since all we do is create a Python virtual environment and install the dependencies from requirements.txt

This codebase is being developed on python version 3.11.6 and tested on MacOS M1 as well as Linux systems. To get started run

1. `python -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. Download the data. See [data/README.md](data/README.md).
5. If on mac (repsp Metal devices), set the env var `PYTORCH_ENABLE_MPS_FALLBACK=1`. This is needed if we train on GPU because of some implementation limitation.

# Structure
- `common` Collection of generic common code.
- `data` contains all the data. As well as some csv files for our classes. The Readme in this folder explains how to download the data.
- `dataloaders` contains two PyTorch datasets for the CINC2020 data. `dataset.py` is our implementation while `datasetThePaperCode.py` is our implementation updated with some of the preprocessing code from the dilated CNN paper.
- `experiments` contains the code for the two experiments: The dialted CNN from the paper and the SWAG experiment.
- `models` contains our model.
- `utils` contains all kinds of utility functionality.

There are many files in `/` which are used to handle the data like flatten the directory structure, splitting it into train and test set etc.

# Reproducability
## Dilated CNN
To run a dilated CNN experiment, you can use `python main_dilatedCNN.py`. You can optionally pass a checkpoint using the `checkpoint_path` argument to continue training from this checkpoint. `main_dilatedCNN.py` reads the config file `experiments/dilated_CNN/config.py` which looks like this:

```py
class Config:
    DATA_DIR = "data/cinc2020_split"
    TEST_DATA_DIR = "data/cinc2020_split/test"
    TRAIN_DATA_DIR = "data/cinc2020_split/train"
    EARLY_STOPPING_EPOCHS = 3
    EXP_NAME = "dilated_CNN"
    LABEL_24 = "data/label_cinc2020_top24.csv"
    LEARNING_RATE = 0.001
    LOAD_OPTIMIZER = True
    MAX_NUM_EPOCHS = 35
    MIN_NUM_EPOCHS = 1
    NUM_FOLDS = 2
    ONLY_EVAL_TEST_SET = False
    OUTPUT_DIR = "outputJan10"
```

whereas

- `DATA_DIR` is the root dir of the data in which the train and test data can be found.
- `TEST_DATA_DIR` points to the test data
- `TRAIN_DATA_DIR`  points to the train data
- `EARLY_STOPPING_EPOCHS` is the amount of epochs for which the challenge score needs to be lower. Then the training is stopped.
- `EXP_NAME` the name of the experiment
- `LABEL_24` path to the 24 classes mapping file
- `LEARNING_RATE` learning rate
- `LOAD_OPTIMIZER` the checkpoint also containes the optimizer state. Set to true if you want to load it.
- `MAX_NUM_EPOCHS` maximal number of epochs to run
- `MIN_NUM_EPOCHS` minimum number of epochs to run
- `NUM_FOLDS` number of folds for CV
- `ONLY_EVAL_TEST_SET` set to true to only evaluate the test set
- `OUTPUT_DIR` dir where all the output is stored

## SWAG
To run a SWAG experiment, you can run `python run_swag.py <config_file_name> <checkpoint_path> <checkpoint_dir>` whereas only `<config_file_name>` isn't optional. Config files can be found in `experiments/SWAG/` dir. You pass the filename as `<config_file_name>`. E.g. to run an SWAG experiment that uses this conifg file ``experiments/SWAG/config_lr_001.py` you'd run `python run_swag.py config_lr_001`. The config files looks like this:

```py
class Config:
    BMA_SAMPLES = 3
    DATA_DIR = "data/cinc2020_split_tiny"
    TEST_DATA_DIR = "data/cinc2020_split_tiny/test"
    TRAIN_DATA_DIR = "data/cinc2020_split_tiny/train"
    DEVIATION_MATRIX_MAX_RANK = 15
    EARLY_STOPPING_EPOCHS = 3
    EXP_NAME = "SWAG"
    LABEL_24 = "data/label_cinc2020_top24.csv"
    LOAD_OPTIMIZER = True
    MAX_NUM_EPOCHS = 5
    NUM_FOLDS = 2  # Training and validation set
    ONLY_EVAL_TEST_SET = False
    OUTPUT_DIR = "output_swag_lr_001"
    SWAG_LEARNING_RATE = 0.001
    SWAG_UPDATE_FREQ = 1
    DEVICE_NAME = "mps"
```

most of the options are the same as in the above section. The difference is:

- `BMA_SAMPLES` Amount of models to sample for Bayesian Model Averaging
- `DEVIATION_MATRIX_MAX_RANK` Amount of model weights to be sampled. This corresponds to `k` in the pseudo code for the SWAG algorithm, see paper mentioned in the report.
- `EARLY_STOPPING_EPOCHS` is actually not used at the moment but can easily be turned on by calling the appropriate function in the metric manager.
- `MAX_NUM_EPOCHS` the amount of epochs that are being run.
- `SWAG_UPDATE_FREQ` How often we should caprute the weights. If it is 1, we capture the weights every epoch. If it is 3, we capture the weights every third epoch etc.
- `DEVICE_NAME` the name of the device we want to run the code on.

## Statistics
While some statistics are reported while training, the statistics that count are those given by the file `evaluate_12ECG_score.py`. You can simply run `python utils/evaluate_12ECG_score.py <test_data_dir> <test_predictions>` whereas the predictions
can be found in the output folder you configured in the corresponding config file of the experiment you run.