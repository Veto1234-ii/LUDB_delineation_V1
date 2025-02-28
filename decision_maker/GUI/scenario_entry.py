from settings import LEADS_NAMES, DELINEATION_LINEWIDTH, POINTS_TYPES_COLORS
from decision_maker.decision_maker_settings import EntriesTypes
from visualisation_utils import plot_one_lead_delineation_on_ax

class ScenarioEntry:
    def __init__(self, color_name, coords_t, lead_name, legend="", text_to_log="", Y_MAX_mV=0.3):
        self.color_name = color_name
        self.coords_t = coords_t
        self.lead_name = lead_name
        self.legend = legend
        self.Y_MAX_mV = Y_MAX_mV

        self.text_to_log = text_to_log

    def draw_to_ax(self, ax):
        plot_one_lead_delineation_on_ax(ax=ax, delineation_t=self.coords_t, color=self.color_name,Y_max=self.Y_MAX_mV, legend=self.legend)


if __name__ == "__main__":
    from settings import LEADS_NAMES, POINTS, FREQUENCY
    from datasets.LUDB_utils import get_some_test_patient_id,  get_one_lead_delineation_by_patient_id, get_signal_by_id_and_lead_mV, get_LUDB_data
    from visualisation_utils import plot_lead_signal_to_ax
    import matplotlib.pyplot as plt

    patient_id = get_some_test_patient_id()

    lead_name = LEADS_NAMES.i
    point_type = POINTS.QRS_PEAK

    LUDB_data = get_LUDB_data()

    delineation = get_one_lead_delineation_by_patient_id(patient_id, LUDB_data=LUDB_data, point_type=point_type, lead_name=lead_name)
    signal_mV = get_signal_by_id_and_lead_mV(patient_id, lead_name=lead_name, LUDB_data=LUDB_data)

    fig, ax = plt.subplots()
    entry = ScenarioEntry(color_name=POINTS_TYPES_COLORS[point_type],
                          coords_t=delineation,
                          lead_name=lead_name,
                          legend="пик QRS",
                          text_to_log="докторская разметка, например",
                          Y_MAX_mV=max(signal_mV))

    entry.draw_to_ax(ax)
    plot_lead_signal_to_ax(signal_mV, ax)
    plt.legend()
    plt.show()