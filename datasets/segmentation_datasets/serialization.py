from datasets.segmentation_datasets.segmentation_dataset import SegmentationDataset
import os
import json
import numpy as np


def load_segmentation_dataset_from_file(name, save_dir="SAVED_DATASETS"):
    save_path = os.path.join(os.path.dirname(__file__), "..", "..", save_dir)
    save_path = os.path.abspath(save_path)  # абсолютный путь

    filename = os.path.join(save_path, f"{name}.json")

    with open(filename, 'r') as file:
        dataset_dict = json.load(file)

    segmentation_dataset = SegmentationDataset(
        name=dataset_dict["name"],
        signals_train=np.array(dataset_dict["signals_train"]),
        signals_test=np.array(dataset_dict["signals_test"]),
        masks_train=np.array(dataset_dict["masks_train"]),
        masks_test=np.array(dataset_dict["masks_test"])
    )
    return segmentation_dataset


def save_segmentation_dataset_to_file(segmentation_dataset, save_dir="SAVED_DATASETS"):
    save_path = os.path.join(os.path.dirname(__file__), "..", "..", save_dir)
    save_path = os.path.abspath(save_path)  # абсолютный путь

    os.makedirs(save_path, exist_ok=True)

    dataset_dict = {
        "name": segmentation_dataset.name,
        "signals_train": segmentation_dataset.signals_train.tolist(),
        "signals_test": segmentation_dataset.signals_test.tolist(),
        "masks_train": segmentation_dataset.masks_train.tolist(),
        "masks_test": segmentation_dataset.masks_test.tolist()
    }

    filename = os.path.join(save_path, f"{segmentation_dataset.name}.json")

    with open(filename, 'w') as file:
        json.dump(dataset_dict, file, indent=4)
