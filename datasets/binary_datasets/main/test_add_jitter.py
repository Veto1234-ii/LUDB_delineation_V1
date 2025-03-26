from datasets.binary_datasets.binary_dataset_creator import create_dataset_from_scratch
from datasets.binary_datasets.serialization import save_binary_dataset_to_file, load_binary_dataset_from_file
from settings import POINTS_TYPES, LEADS_NAMES, PATH_TO_LUDB

from pathlib import Path
import json


if __name__ == "__main__":
    from datasets.GUI import UIBinaryDataset

    # Открываем LUDB
    path_to_dataset = Path(PATH_TO_LUDB)
    with open(path_to_dataset, 'r') as file:
        LUDB_dataset = json.load(file)

    # Составляем свой датасет
    binary_dataset = create_dataset_from_scratch(point_name=POINTS_TYPES.QRS_PEAK,
                                                 radius=200,
                                                 lead_name=LEADS_NAMES.i,
                                                 LUDB_dataset=LUDB_dataset,
                                                 dataset_size=50
                                                 )


    # Анализ датасета перед добавлением джиттер-примеров
    print(f"Исходный размер тренировочной выборки: {len(binary_dataset.signals_train)}")
    print(f"Количество положительных примеров: {sum(binary_dataset.labels_train)}")
    print(f"Количество негативных примеров: {len(binary_dataset.labels_train) - sum(binary_dataset.labels_train)}")
    print("Визуализация ДО добавления джиттер-примеров:")
    # Листаем его в UI с визуализацией картинок
    binary_dataset_visualizator = UIBinaryDataset(binary_dataset)

    # Добавление джиттер-примеров
    print("\nДобавляем 100 джиттер-примеров")
    binary_dataset.add_jitter(num_of_jitter_examples=100)

    # Анализ датасета после добавлением джиттер-примеров
    print(f"Исходный размер тренировочной выборки: {len(binary_dataset.signals_train)}")
    print(f"Количество положительных примеров: {sum(binary_dataset.labels_train)}")
    print(f"Количество негативных примеров: {len(binary_dataset.labels_train) - sum(binary_dataset.labels_train)}")

    # Визуализация после добавления
    print("\nВизуализация ПОСЛЕ добавления джиттер-примеров:")
    binary_dataset_visualizator = UIBinaryDataset(binary_dataset)