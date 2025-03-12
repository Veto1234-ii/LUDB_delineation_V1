from datasets.binary_datasets.binary_dataset import BinaryDataset
import os
import json
import numpy as np

def load_binary_dataset_from_file(name, save_dir="SAVED_DATASETS"):
    # TODO составляем имя файла как name + расширение
    # TODO возвращаем объект BinaryDataset

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
        labels_test=np.array(dataset_dict["labels_test"])
    )
    return binary_dataset

def save_binary_dataset_to_file(binary_dataset, save_dir="SAVED_DATASETS"):
    # TODO сохранение в файл с именем binary_dataset.name + расширение

    save_path = os.path.join(os.path.dirname(__file__), "..", "..", save_dir)
    save_path = os.path.abspath(save_path)  # абсолютный путь

    # если нет папки
    os.makedirs(save_path, exist_ok=True)

    dataset_dict = {
        "name": binary_dataset.name,
        "signals_train": binary_dataset.signals_train.tolist(),  # np array в список
        "signals_test": binary_dataset.signals_test.tolist(),
        "labels_train": binary_dataset.labels_train.tolist(),
        "labels_test": binary_dataset.labels_test.tolist()
    }

    filename = os.path.join(save_path, f"{binary_dataset.name}.json")

    with open(filename, 'w') as file:
        json.dump(dataset_dict, file, indent=4)
