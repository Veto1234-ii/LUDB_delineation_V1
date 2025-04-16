from datasets.binary_datasets.binary_dataset import BinaryDataset
import os
import json
import numpy as np

def load_binary_dataset_from_file(name, save_dir="SAVED_DATASETS"):
    """
        Args:
            name: имя binary_dataset
            save_dir: место сохранённого binary_dataset

        Returns: (BinaryDataset) датасет бинарной классификации
    """

    save_path = os.path.join(os.path.dirname(__file__), "..", "..", save_dir)
    save_path = os.path.abspath(save_path)  # абсолютный путь

    filename = os.path.join(save_path, f"{name}.json")

    with open(filename, 'r') as file:
        dataset_dict = json.load(file)

    binary_dataset = BinaryDataset(
        name=dataset_dict["name"],
        signals_train=np.array(dataset_dict["signals_train"]),
        signals_test=np.array(dataset_dict["signals_test"]),
        labels_train=np.array(dataset_dict["labels_train"]),
        labels_test=np.array(dataset_dict["labels_test"]),
        full_signals=np.array(dataset_dict["full_signals"]),
        delineation_on_full_signals=np.array(dataset_dict["delineation_on_full_signals"]),
        lead_name = dataset_dict["lead_name"],
        point_name = dataset_dict["point_name"],
        radius = dataset_dict["radius"]
    )
    return binary_dataset

def save_binary_dataset_to_file(binary_dataset, save_dir="SAVED_DATASETS"):
    """
        Args:
            binary_dataset: (BinaryDataset)  binary_dataset который сохраняется
            save_dir: место для сохранения

    """

    save_path = os.path.join(os.path.dirname(__file__), "..", "..", save_dir)
    save_path = os.path.abspath(save_path)  # абсолютный путь

    # если нет папки
    os.makedirs(save_path, exist_ok=True)

    dataset_dict = {
        "name": binary_dataset.name,
        "signals_train": binary_dataset.signals_train.tolist(),  # np array в список
        "signals_test": binary_dataset.signals_test.tolist(),
        "labels_train": binary_dataset.labels_train.tolist(),
        "labels_test": binary_dataset.labels_test.tolist(),
        "full_signals": binary_dataset.full_signals.tolist(),
        "delineation_on_full_signals": binary_dataset.delineation_on_full_signals.tolist(),
        "lead_name" : binary_dataset.lead_name,
        "point_name" : binary_dataset.point_name,
        "radius" : binary_dataset.radius
    }

    filename = os.path.join(save_path, f"{binary_dataset.name}.json")

    with open(filename, 'w') as file:
        json.dump(dataset_dict, file, indent=4)
