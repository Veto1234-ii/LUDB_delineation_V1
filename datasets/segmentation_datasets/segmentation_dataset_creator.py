from datasets.segmentation_datasets.segmentation_dataset import SegmentationDataset
from datasets.LUDB_utils import get_signal_by_id_and_lead_mV
from datasets.LUDB_utils import get_test_and_train_ids
from datasets.LUDB_utils.get_delineation_by_patient_id import get_full_wave_delineation_by_patient_id
from settings import FREQUENCY

import numpy as np
import random

def create_segmentation_dataset(wave_type, radius, lead_name, LUDB_dataset, dataset_size=1000):
    """

    Args:
        wave_type: тип волны, брать в WAVES_TYPES
        radius: (int) радиус окна вокруг случайного центра
        lead_name: имя отведения, брать его в LEADS_NAMES
        LUDB_dataset: загруженный из json файл датасета LUDB
        dataset_size: сколько экземпляров хотим в нашем датасете (это длина теста + длина трейна)

    Returns: (SegmentationDataset) датасет для сегментации

    """

    # Разделение пациентов на обучающую и тестовую выборки
    train_ids, test_ids = get_test_and_train_ids(LUDB_dataset)

    dataset_name = f"SegmentationDataset_{dataset_size}_{lead_name}_{wave_type}_{radius}"

    train_size = int(dataset_size * 0.7)  # 70% от dataset_size
    test_size = dataset_size - train_size  # 30% -//-

    signals_train, masks_train = [], []  # трейн микродатасета
    signals_test, masks_test = [], []  # тест микродатасета

    # Создание трейна микродатасета
    while len(signals_train) < train_size:
        patient_id = random.choice(train_ids)
        signal = get_signal_by_id_and_lead_mV(patient_id, lead_name, LUDB_dataset)
        wave_triplets = get_full_wave_delineation_by_patient_id(patient_id, LUDB_dataset, lead_name, wave_type)

        # маска для всего сигнала
        full_mask = np.zeros(len(signal), dtype=int)
        for triplet in wave_triplets:
            start = int(triplet[0] * FREQUENCY)
            end = int(triplet[2] * FREQUENCY)
            full_mask[start:end + 1] = 1

        random_index = random.randint(radius, len(signal) - radius)

        signals_train.append(signal[random_index - radius:random_index + radius])
        masks_train.append(full_mask[random_index - radius:random_index + radius])

    # Создание теста микродатасета
    while len(signals_test) < test_size:
        patient_id = random.choice(test_ids)
        signal = get_signal_by_id_and_lead_mV(patient_id, lead_name, LUDB_dataset)
        wave_triplets = get_full_wave_delineation_by_patient_id(patient_id, LUDB_dataset, lead_name, wave_type)

        # маска для всего сигнала
        full_mask = np.zeros(len(signal), dtype=int)
        for triplet in wave_triplets:
            start = int(triplet[0] * FREQUENCY)
            end = int(triplet[2] * FREQUENCY)
            full_mask[start:end + 1] = 1

        random_index = random.randint(radius, len(signal) - radius)

        signals_test.append(signal[random_index - radius:random_index + radius])
        masks_test.append(full_mask[random_index - radius:random_index + radius])

    segmentation_dataset = SegmentationDataset(
        name=dataset_name,
        signals_train=np.array(signals_train),
        signals_test=np.array(signals_test),
        masks_train=np.array(masks_train),
        masks_test=np.array(masks_test)
    )

    segmentation_dataset.shuffle()
    return segmentation_dataset
