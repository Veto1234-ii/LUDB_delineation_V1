from idlelib.search import SearchDialog

from settings import WAVES_TYPES_COLORS, LEADS_NAMES

class SearchInterval:
    def __init__(self, t_start, t_end, lead_name):
        """
        Объект сцены, инкапсулирующий интервал, в котором мы планируем что-то искать.
        Имеет начало и конец.
        Args:
            t_start: (float) начало интервала в секундах
            t_end: (float) конец интервала в секундах

            lead_name: константа обозн. отведение
        """

        self.t_start = t_start
        self.t_end = t_end
        self.lead_name = lead_name

        self.id_in_scene = None  # Автоматически назначается сценой

    def draw(self, ax):
        ax.axvspan(self.t_start, self.t_end, color='black', alpha=0.2, hatch='///', edgecolor='red', linewidth=1)


if __name__ == "__main__":
    from settings import LEADS_NAMES
    from datasets.LUDB_utils import get_some_test_patient_id, get_signal_by_id_and_lead_mV, get_LUDB_data
    from visualisation_utils import plot_lead_signal_to_ax
    import matplotlib.pyplot as plt

    # Рисуем сигнал
    LUDB_data = get_LUDB_data()
    patient_id = get_some_test_patient_id()
    lead_name = LEADS_NAMES.i
    signal_mV = get_signal_by_id_and_lead_mV(patient_id, lead_name=lead_name, LUDB_data=LUDB_data)
    fig, ax = plt.subplots()
    plot_lead_signal_to_ax(signal_mV=signal_mV, ax=ax)

    # Придумываем интервал и рисуем
    search_interval = SearchInterval(t_start=1, t_end=2.1, lead_name=LEADS_NAMES.i)
    search_interval.draw(ax)

    # показываем итог: сигнал и облако активаций поверх него
    plt.show()
