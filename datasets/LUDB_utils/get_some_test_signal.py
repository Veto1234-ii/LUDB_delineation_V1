from settings import PATH_TO_LUDB, LEADS_NAMES

import json
from pathlib import Path


# Чтоб преобразовать x к времени, надо делить на 500.
# Чтоб преобразовать y к mV надо делить на 1000.
# Метод возвращаем не преобразованный сигнал:

def get_some_test_signal():
    # Откроем датасет LUDB
    path_to_dataset = Path(PATH_TO_LUDB)
    with open(path_to_dataset, 'r') as file:
        LUDB_data = json.load(file)

    # возьмем id-шник i-того  пациента датасета
    i = 35
    patient_ids = list(LUDB_data.keys())
    patient_id = patient_ids[i]

    # Выберем интересующее нас отведение
    lead_name = LEADS_NAMES.i

    # вытащим сам сигнал из этого отведения этого пациента
    signal_one_lead=  LUDB_data[patient_id]['Leads'][lead_name]['Signal']

    return signal_one_lead


