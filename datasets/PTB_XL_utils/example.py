from get_id_of_records_with_diagnosis import get_id_of_records_with_diagnosis_normal
from get_signals_by_id_and_leads import get_signals_by_id_and_leads
from settings import LEADS_NAMES
from vizualizator_from_PTB_XL import visualize_ecg_signals


leads = [LEADS_NAMES.i, LEADS_NAMES.ii, LEADS_NAMES.v3]

normal_id = get_id_of_records_with_diagnosis_normal()

signals = get_signals_by_id_and_leads(normal_id[3], leads, fs=500)

visualize_ecg_signals(signals, leads)