from datasets.binary_datasets.binary_dataset import BinaryDataset
from datasets.LUDB_utils import get_signal_by_id_and_lead_mV, get_one_lead_delineation_by_patient_id, \
    get_signal_by_id_and_lead_mkV
from datasets.LUDB_utils import get_test_and_train_ids
from settings import FREQUENCY

import numpy as np
import random

def create_dataset_from_scratch(point_name, radius, lead_name, LUDB_dataset, dataset_size = 10000):
    """

    Args:
        point_name: тип точки, один из 9, брать в POINTS_TYPES
        radius: (int) радус окна вокруг точки, число от 1 до MAX_SIGNAL_LEN/2
        lead_name: имя отведения, брать его в LEADS_NAMES
        LUDB_dataset: загруженый из json файл датасета LUDB
        dataset_size: сколько экземпляров хотим в нашем датасете (это длина теста + длина трейна)

    Returns: (BinaryDataset) датасет бинарной классификации

    """

    # Разделение пациентов на обучающую и тестовую выборки
    train_ids, test_ids = get_test_and_train_ids(LUDB_dataset)

    dataset_name = f"BinaryDataset_{dataset_size}_{lead_name}_{point_name.name}_{radius}"

    train_size = dataset_size

    signals_train, labels_train = [], []  # трейн микродатасета
    signals_test, labels_test = [], []  # тест

    # для джиттер-примеров
    full_signals = []
    delineation_on_full_signals = []

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
                    full_signals.append(signal)
                    delineation_on_full_signals.append(selected_point_name)

        # разметка не в центре
        else:
            random_index = np.random.randint(radius, len(signal) - radius)
            if (random_index / FREQUENCY) not in delineation_by_point_name:
                if (random_index - radius >= 0) and (random_index + radius < len(signal)):
                    signals_train.append(signal[random_index - radius:random_index + radius])
                    labels_train.append(0)
                    full_signals.append(signal)
                    delineation_on_full_signals.append(random_index)

    # Создание теста  - это сигналы полной длины + верные ответы для них, представляющие собой массивы координакт разметки
    for id_ in test_ids:
        signals_test.append(get_signal_by_id_and_lead_mkV(id_, lead_name, LUDB_dataset))
        labels_test.append(
            [int(FREQUENCY * i) for i in get_one_lead_delineation_by_patient_id(id_, LUDB_dataset, lead_name, point_name)])

    binary_dataset = BinaryDataset(
        name=dataset_name,
        signals_train=np.array(signals_train),
        signals_test=np.array(signals_test),
        labels_train=np.array(labels_train),
        labels_test=labels_test,
        full_signals=np.array(full_signals),
        delineation_on_full_signals=np.array(delineation_on_full_signals)
    )

    binary_dataset.shuffle()
    return binary_dataset
