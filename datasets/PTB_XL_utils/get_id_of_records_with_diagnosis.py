import pandas as pd
import ast
from paths import PATH_TO_PTB_XL


def get_id_of_records_with_diagnosis_normal(metadata_path = PATH_TO_PTB_XL / "ptbxl_database.csv", min_confidence=100):
    """
        Args:
            metadata_path (str): Путь к метаданным датасета
            min_confidence (int): значение NORM

        Returns:
            ids (list) : список id где NORM
    """
    data_frame = pd.read_csv(metadata_path)

    # bool заполнение
    is_normal = []
    for codes in data_frame['scp_codes']:
        try:
            scp_dict = ast.literal_eval(codes)
            is_normal.append('NORM' in scp_dict and scp_dict['NORM'] >= min_confidence)
        except:
            is_normal.append(False)

    # остаётся где is_normal=True и id
    return data_frame[is_normal]['ecg_id'].tolist()
