from pathlib import Path
import json

from neural_networks.neural_networks_helpers.helpers_CNN import get_F1_of_one_CNN

from neural_networks.neural_networks_models.CNN import save_model
from datasets.binary_datasets.binary_dataset_creator import create_dataset_from_scratch
from settings import POINTS_TYPES, LEADS_NAMES, PATH_TO_LUDB, LEADS_NAMES_ORDERED

from neural_networks.neural_networks_helpers import get_appliable


def train_one_net(point_type, lead_name):
    # Открываем LUDB
    path_to_dataset = Path(PATH_TO_LUDB)
    with open(path_to_dataset, 'r') as file:
        LUDB_dataset = json.load(file)

    # Составляем свой датасет
    binary_dataset = create_dataset_from_scratch(point_name=point_type,
                                                 radius=250,
                                                 lead_name=lead_name,
                                                 LUDB_dataset=LUDB_dataset
                                                 )

    # Обучаем и сохраняем модель
    save_model(binary_dataset, point_type, lead_name)

def train_bunch_of_nets():
    # Открываем LUDB
    path_to_dataset = Path(PATH_TO_LUDB)
    with open(path_to_dataset, 'r') as file:
        LUDB_dataset = json.load(file)

    all_points = [p for p in POINTS_TYPES]
    print(all_points)
    for lead_name in LEADS_NAMES_ORDERED:
        for point_type in all_points:

            # Составляем свой датасет
            binary_dataset = create_dataset_from_scratch(point_name=point_type,
                                                         radius=250,
                                                         lead_name=lead_name,
                                                         LUDB_dataset=LUDB_dataset
                                                         )

            # Обучаем и сохраняем модель
            save_model(binary_dataset, point_type, lead_name)


if __name__ == "__main__":

    # Обучение одной сети
    # train_one_net(POINTS_TYPES.T_PEAK, LEADS_NAMES.i)

    #train_bunch_of_nets()
    
    path_to_dataset = Path(PATH_TO_LUDB)
    with open(path_to_dataset, 'r') as file:
        LUDB_dataset = json.load(file)

    # Составляем свой датасет
    binary_dataset = create_dataset_from_scratch(point_name=POINTS_TYPES.QRS_PEAK,
                                                 radius=250,
                                                 lead_name=LEADS_NAMES.i,
                                                 LUDB_dataset=LUDB_dataset
                                                 )
    
    model = get_appliable('BinaryDataset_10000_i_QRS_PEAK_250_0319_011511')
    
    F1, mean_err = get_F1_of_one_CNN(model, binary_dataset.get_test()[0], binary_dataset.get_test()[1], threshold=0.8, tolerance=25)

