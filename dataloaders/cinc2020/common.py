import pandas as pd

from experiments.dilated_CNN.config import Config

mappings = pd.read_csv(Config.LABEL_24, delimiter=",")
labels_map = mappings["SNOMED CT Code"].values
