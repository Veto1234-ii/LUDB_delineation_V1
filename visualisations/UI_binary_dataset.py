from datasets.binary_datasets import BinaryDataset

import numpy as np

class UIBinaryDataset:
    def __init__(self, binary_dataset):
        train_data = binary_dataset.get_train()

