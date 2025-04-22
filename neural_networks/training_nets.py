import os
from pathlib import Path
import json

import torch

from datasets.segmentation_datasets.segmentation_dataset_creator import create_segmentation_dataset
from neural_networks.neural_networks_helpers.helpers_CNN.F1_of_CNN import get_F1_segmentation_of_one_CNN
from neural_networks.neural_networks_models.CNN import save_model
from neural_networks.neural_networks_models.CNN_2 import save_model_2
from datasets.binary_datasets.binary_dataset_creator import create_dataset_from_scratch
from paths import PATH_TO_LUDB
from settings import POINTS_TYPES, LEADS_NAMES, LEADS_NAMES_ORDERED, WAVES_TYPES, TOLERANCE


def train_one_net(point_type, lead_name):
    # Открываем LUDB
    path_to_dataset = PATH_TO_LUDB
    with open(path_to_dataset, 'r') as file:
        LUDB_dataset = json.load(file)

    # Составляем свой датасет
    binary_dataset = create_dataset_from_scratch(point_name=point_type,
                                                 radius=500,
                                                 lead_name=lead_name,
                                                 LUDB_dataset=LUDB_dataset,
                                                 dataset_size=5000
                                                 )

    binary_dataset_qrs = create_dataset_from_scratch(point_name=POINTS_TYPES.QRS_PEAK,
                                                 radius=600,
                                                 lead_name=lead_name,
                                                 LUDB_dataset=LUDB_dataset,
                                                 dataset_size=0
                                                 )

    # step = 10
    # binary_dataset.add_jitter(num_of_jitter_examples=200, jitter_range=(50, 29, -step))  # 1 - 2000

    # # Составляем свой датасет
    # segmentation_dataset = create_segmentation_dataset(
    #     wave_type=WAVES_TYPES.P,
    #     radius=250,
    #     lead_name=lead_name,
    #     LUDB_dataset=LUDB_dataset
    # )

    # Составляем свой датасет
    binary_dataset_seg = create_dataset_from_scratch(point_name=point_type,
                                                 radius=int(600+1.5*TOLERANCE),
                                                 lead_name=lead_name,
                                                 LUDB_dataset=LUDB_dataset,
                                                 dataset_size=1000
                                                 )

    # Обучаем и сохраняем модель
    save_model(binary_dataset, point_type, lead_name, 1)
    save_model_2(binary_dataset, point_type, lead_name, 1, binary_dataset_seg, input_size=1000, binary_dataset_qrs=binary_dataset_qrs)


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
    # train_one_net(POINTS_TYPES.P_PEAK, LEADS_NAMES.ii)
    # train_one_net(POINTS_TYPES.QRS_PEAK, LEADS_NAMES.ii)
    train_one_net(POINTS_TYPES.T_PEAK, LEADS_NAMES.ii)
    # train_one_net(POINTS_TYPES.P_START, LEADS_NAMES.ii)
    # train_one_net(POINTS_TYPES.P_END, LEADS_NAMES.ii)
    # train_one_net(POINTS_TYPES.QRS_START, LEADS_NAMES.ii)
    # train_one_net(POINTS_TYPES.QRS_END, LEADS_NAMES.ii)
    # train_one_net(POINTS_TYPES.T_START, LEADS_NAMES.ii)
    # train_one_net(POINTS_TYPES.T_END, LEADS_NAMES.ii)
    #
    # train_one_net(POINTS_TYPES.P_PEAK, LEADS_NAMES.iii)
    # train_one_net(POINTS_TYPES.QRS_PEAK, LEADS_NAMES.iii)
    # train_one_net(POINTS_TYPES.T_PEAK, LEADS_NAMES.iii)
    # train_one_net(POINTS_TYPES.P_START, LEADS_NAMES.iii)
    # train_one_net(POINTS_TYPES.P_END, LEADS_NAMES.iii)
    # train_one_net(POINTS_TYPES.QRS_START, LEADS_NAMES.iii)
    # train_one_net(POINTS_TYPES.QRS_END, LEADS_NAMES.iii)
    # train_one_net(POINTS_TYPES.T_START, LEADS_NAMES.iii)
    # train_one_net(POINTS_TYPES.T_END, LEADS_NAMES.iii)
    #
    # train_one_net(POINTS_TYPES.P_PEAK, LEADS_NAMES.i)
    # train_one_net(POINTS_TYPES.QRS_PEAK, LEADS_NAMES.i)
    # train_one_net(POINTS_TYPES.T_PEAK, LEADS_NAMES.i)
    # train_one_net(POINTS_TYPES.P_START, LEADS_NAMES.i)
    # train_one_net(POINTS_TYPES.P_END, LEADS_NAMES.i)
    # train_one_net(POINTS_TYPES.QRS_START, LEADS_NAMES.i)
    # train_one_net(POINTS_TYPES.QRS_END, LEADS_NAMES.i)
    # train_one_net(POINTS_TYPES.T_START, LEADS_NAMES.i)
    # train_one_net(POINTS_TYPES.T_END, LEADS_NAMES.i)

    # Обучение нескольких сетей
    #train_bunch_of_nets()

