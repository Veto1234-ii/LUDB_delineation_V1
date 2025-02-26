from settings import LEADS_NAMES, LEADS_NAMES_ORDERED

def get_signal_by_id_and_lead(patient_id, lead_name, LUDB_data):
    patinet_data = LUDB_data[patient_id]
    # TODO Взять из patinet_data сигнал отведения с именем lead_name
    return signal

def get_all_signals_by_id(patient_id, LUDB_data):
    signals_list = []
    for lead_name in LEADS_NAMES_ORDERED:
        lead_signal =  get_signal_by_id_and_lead(patient_id, lead_name, LUDB_data)
        signals_list.append(lead_signal)
    return signals_list, LEADS_NAMES_ORDERED
