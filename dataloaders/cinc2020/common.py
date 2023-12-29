import pandas as pd

mappings = pd.read_csv("data/cinc2020/label_cinc2020_top24.csv", delimiter=",")
labels_map = mappings["SNOMED CT Code"].values
