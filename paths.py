from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SAVED_NETS_PATH = PROJECT_ROOT / "SAVED_NETS"  # У себя писать from paths import SAVED_NETS_PATH
PATH_TO_LUDB = PROJECT_ROOT / "LUDB\\ecg_data_200.json"
PATH_TO_LUDB_DIAGNOSIS_DICT = PROJECT_ROOT / "LUDB\\diagnosis.json"