from binary_dataset import BinaryDataset
from binary_dataset_creator import create_dataset_from_scratch
from serialization import save_binary_dataset_to_file, load_binary_dataset_from_file
from settings import POINTS, LEADS_NAMES, PATH_TO_LUDB

from pathlib import Path
import json


if __name__ == "__main__":
    from visualisations import BinaryDatasetVis

    # Открываем LUDB
    path_to_dataset = Path(PATH_TO_LUDB)
    with open(path_to_dataset, 'r') as file:
        LUDB_dataset = json.load(file)

    # Составляем свой датасет
    binary_dataset = create_dataset_from_scratch(point_name=POINTS.QRS_PEAK,
                                radius=200,
                                lead_name=LEADS_NAMES.i,
                                patient_ids=list(LUDB_dataset.keys()),
                                LUDB_dataset=LUDB_dataset
                                )
    # сохраняем в файл
    name = binary_dataset.name
    save_binary_dataset_to_file(binary_dataset=binary_dataset)
    del binary_dataset

    # загружаем из файла
    binary_dataset = load_binary_dataset_from_file(name=name)

    # Листаем его в UI с визуализацией картинок
    binary_dataset_visualizator = BinaryDatasetVis(binary_dataset)