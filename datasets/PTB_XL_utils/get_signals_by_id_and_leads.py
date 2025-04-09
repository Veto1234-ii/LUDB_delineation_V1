import pandas as pd
import wfdb
import ast
from paths import PATH_TO_PTB_XL


def get_signals_by_id_and_leads(ecg_id, leads, fs=500, path=PATH_TO_PTB_XL):
    """
    Args:
        ecg_id (int): ID записи из ptbxl_database.csv
        leads (list): Список отведений (можно любым регистром)
        fs (int): Частота дискретизации (100 или 500)
        path (str): Путь к датасету

    Returns:
        signals (list) : список сигналов по id и отведениям
    """
    # 1. Загружаем метаданные
    metadata_path = path / "ptbxl_database.csv"
    Y = pd.read_csv(metadata_path, index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)

    # существование записи
    if ecg_id not in Y.index:
        raise ValueError(f"ECG ID {ecg_id} не найден")

    # имя файла из метаданных
    record = Y.loc[ecg_id]
    filename = record.filename_lr if fs == 100 else record.filename_hr

    # данные из records100/500
    full_path = path / filename
    signals, meta = wfdb.rdsamp(full_path)

    # отбор нужных отведений из meta['sig_name'] = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    lead_indices = []
    valid_leads = []
    for lead in leads:
        lead = lead.upper()
        if lead in meta['sig_name']:
            idx = meta['sig_name'].index(lead)
            lead_indices.append(idx)
            valid_leads.append(lead)

    if not lead_indices:
        raise ValueError("Нет совпадений по отведениям")

    # извлечение нужных сигналов
    ecg_data = signals[:, lead_indices].T

    return ecg_data.tolist()
