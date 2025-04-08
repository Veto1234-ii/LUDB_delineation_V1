from datasets.GUI.UI_show_ECG_from_LUDB import UI


def visualize_ecg_signals(signals, leads):
    leads_names = [lead.upper() for lead in leads]

    UI(signals=signals, leads_names=leads_names)