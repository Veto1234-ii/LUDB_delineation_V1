from datasets.binary_datasets import BinaryDataset

import numpy as np

# Листалка примеров из датасета. Нужна, чтобы зрительно ознакомиться с собранным датасетом.
class UIBinaryDataset:
    def __init__(self, binary_dataset):
        train_data = binary_dataset.get_train()
        #TODO

