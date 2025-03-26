from datasets.segmentation_datasets.segmentation_dataset_creator import create_segmentation_dataset
from datasets.segmentation_datasets.serialization import save_segmentation_dataset_to_file, load_segmentation_dataset_from_file
from settings import WAVES_TYPES, LEADS_NAMES
from paths import PATH_TO_LUDB

from pathlib import Path
import json

if __name__ == "__main__":
    # Открываем LUDB
    path_to_dataset = Path(PATH_TO_LUDB)
    with open(path_to_dataset, 'r') as file:
        LUDB_dataset = json.load(file)

    # Составляем свой датасет
    segmentation_dataset = create_segmentation_dataset(
        wave_type=WAVES_TYPES.T,
        radius=200,
        lead_name=LEADS_NAMES.ii,
        LUDB_dataset=LUDB_dataset
    )

    # сохраняем в файл
    name = segmentation_dataset.get_name()
    save_segmentation_dataset_to_file(segmentation_dataset, save_dir="SAVED_DATASETS")
    del segmentation_dataset

    # загружаем из файла
    segmentation_dataset = load_segmentation_dataset_from_file(name, save_dir="SAVED_DATASETS")

    # Визуализация (если есть GUI)
    # segmentation_dataset_visualizator = UIBinaryDataset(segmentation_dataset)