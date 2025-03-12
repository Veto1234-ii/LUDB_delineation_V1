from datasets.binary_datasets.binary_dataset import BinaryDataset
from datasets.LUDB_utils import get_signal_by_id_and_lead_mV, get_one_lead_delineation_by_patient_id
from datasets.LUDB_utils import get_test_and_train_ids
from settings import FREQUENCY

import numpy as np
import random

def create_dataset_from_scratch(point_name, radius, lead_name, LUDB_dataset):

    # TODO составляем имя датасета как "Бинарный датасет " + размер датасета, название отведения, какую точку искали , радиус
    # в LUDB_utils взять id-шники трейновых и тестовых пациентов
    # TODO на первой грппе пациентов собираем трейн, на второй тест  - уже нашего микродатасета
    # TODO заполняем объект BinaryDataset и возвращаем его

    dataset_size = 1000

    # Разделение пациентов на обучающую и тестовую выборки
    train_ids, test_ids = get_test_and_train_ids(LUDB_dataset)

    dataset_name = f"BinaryDataset_{dataset_size}_{lead_name}_{point_name.name}_{radius}"

    train_size = int(dataset_size * 0.7)  # 70% от dataset_size
    test_size = dataset_size - train_size  # 30% -//-

    signals_train, labels_train = [], []  # трейн микродатасета
    signals_test, labels_test = [], []  # тест микродатасета

    # Создание трейна микродатасета
    while len(signals_train) < train_size:
        patient_id = random.choice(train_ids)
        delineation_by_point_name = get_one_lead_delineation_by_patient_id(patient_id, LUDB_dataset, lead_name, point_name)
        signal = get_signal_by_id_and_lead_mV(patient_id, lead_name, LUDB_dataset)

        # разметка в центре
        if len(signals_train) < train_size // 2:
            if len(delineation_by_point_name) > 0:
                selected_point_name = int(random.choice(delineation_by_point_name) * FREQUENCY)
                if (selected_point_name - radius >= 0) and (selected_point_name + radius < len(signal)):
                    signals_train.append(signal[selected_point_name - radius:selected_point_name + radius])
                    labels_train.append(1)
        # разметка не в центре
        else:
            random_index = np.random.randint(radius, len(signal) - radius)
            if (random_index / FREQUENCY) not in delineation_by_point_name:
                if (random_index - radius >= 0) and (random_index + radius < len(signal)):
                    signals_train.append(signal[random_index - radius:random_index + radius])
                    labels_train.append(0)

    # Создание теста микродатасета
    while len(signals_test) < test_size:
        patient_id = random.choice(test_ids)
        delineation_by_point_name = get_one_lead_delineation_by_patient_id(patient_id, LUDB_dataset, lead_name, point_name)
        signal = get_signal_by_id_and_lead_mV(patient_id, lead_name, LUDB_dataset)

        # разметка в центре
        if len(signals_test) < test_size // 2:
            if len(delineation_by_point_name) > 0:
                selected_point_name = int(random.choice(delineation_by_point_name) * FREQUENCY)
                if (selected_point_name - radius >= 0) and (selected_point_name + radius < len(signal)):
                    signals_test.append(signal[selected_point_name - radius:selected_point_name + radius])
                    labels_test.append(1)
        # разметка не в центре
        else:
            random_index = np.random.randint(radius, len(signal) - radius)
            if (random_index / FREQUENCY) not in delineation_by_point_name:
                if (random_index - radius >= 0) and (random_index + radius < len(signal)):
                    signals_test.append(signal[random_index - radius:random_index + radius])
                    labels_test.append(0)

    binary_dataset = BinaryDataset(name=dataset_name, signals_train=np.array(signals_train), signals_test=np.array(signals_test), labels_train=np.array(labels_train), labels_test=np.array(labels_test))

    binary_dataset.shuffle()
    return binary_dataset
