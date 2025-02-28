from settings import FREQUENCY, DELINEATION_LINEWIDTH

import matplotlib.pyplot as plt
import random

def plot_one_lead_delineation_on_ax(ax, delineation_t, Y_max, color=None, legend="легенда не заполнена"):
    if color is None:
        color = (random.random(), random.random(), random.random())
    for x in delineation_t:
        ax.axvline(x=x, ymax=Y_max, ymin=0, color=color, linewidth=DELINEATION_LINEWIDTH,  label=legend if x == delineation_t[0] else "")


if __name__ == "__main__":
    from settings import LEADS_NAMES, POINTS, FREQUENCY, POINTS_TYPES_COLORS
    from datasets.LUDB_utils import get_some_test_patient_id,  get_one_lead_delineation_by_patient_id, get_signal_by_id_and_lead_mV, get_LUDB_data
    from visualisation_utils import plot_lead_signal_to_ax
    import matplotlib.pyplot as plt

    patient_id = get_some_test_patient_id()
    LUDB_data = get_LUDB_data()

    # извлекаем и отрисовываем сигнал
    lead_name = LEADS_NAMES.i
    signal_mV = get_signal_by_id_and_lead_mV(patient_id, lead_name=lead_name, LUDB_data=LUDB_data)

    fig, ax = plt.subplots()
    plot_lead_signal_to_ax(signal_mV=signal_mV, ax=ax)

    # извлекаем и отрисовываем размету
    point_type = POINTS.QRS_PEAK
    delineation = get_one_lead_delineation_by_patient_id(patient_id, LUDB_data=LUDB_data, point_type=point_type,
                                                         lead_name=lead_name)

    plot_one_lead_delineation_on_ax(ax, delineation_t=delineation, Y_max=max(signal_mV), color=POINTS_TYPES_COLORS[point_type], legend="QRS-пик")

    ax.legend()
    plt.show()