# Introduction
Group project for the Deep Learning course in HS 2023
Working on AI in ECG, in particular trying to introduce SWAG method and compare results with vanilla-SGD 

This repo is aiming at two main things:
1. Craft ideas and brainstorm solutions, visualize data
2. Train a model and return the results as part of the project

# Structure
Since we have several datasets and probably utilize several models, this codebase follows a modular approach utilizing [Python modules](https://docs.python.org/3/tutorial/modules.html)

First we have the data in the `data` directory. Each subdirectory is considered its own dataset. For each dataset we provide a [PyTorch dataset and dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). Secondly we have the models classes in the `models` directory. And third we have the experiments in the `experiment` folder. Each experiment can utilize any of the dataset and any of the models.

Note that the data in the `data` folder isn't commited to the repository because of its size, which is why it should be added to the .gitignore.

- `data`: This directory contains the actual data as well as the correspoding [PyTorch dataset and dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- `models`: Contains the code for the actual neural networks. Each subfolder represents a model.
- `experiments`: Contains the code for the actual experiments we run i.e. the code for training and evaluation.

# TODO:
- Create setup script that downloads data etc.