from settings import WAVES_TYPES_COLORS

class DelineationInterval:
    def __init__(self, t_start, t_end, wave_type, label=None):
        """
        Объект сцены, инкапсулирующий интервал разметки, у интервала определенный тип, начало и конец
        Args:
            t_start: (float) начало интервала в секундах
            t_end: (float) конец интервала в секундах
            wave_type: тип интервала (см. settings)
            label: (str) пояснение для лога и легенды, если нужно
        """
        self.label = label
        self.t_start = t_start
        self.t_end = t_end
        self.interval_type = wave_type

    def contain(self, t):
        return t<= self.t_end and t>= self.t_start

    def draw(self, ax):
        color = WAVES_TYPES_COLORS[self.interval_type]
        ax.axvspan(self.t_start, self.t_end, color=color, alpha=0.3, label=self.label)


if __name__ == "__main__":
    from settings import LEADS_NAMES, FREQUENCY, WAVES_TYPES
    from datasets.LUDB_utils import get_some_test_patient_id, get_signal_by_id_and_lead_mV, get_LUDB_data
    from visualisation_utils import plot_lead_signal_to_ax
    import matplotlib.pyplot as plt
    import math

    # Рисуем сигнал
    LUDB_data = get_LUDB_data()
    patient_id = get_some_test_patient_id()
    lead_name = LEADS_NAMES.i
    signal_mV = get_signal_by_id_and_lead_mV(patient_id, lead_name=lead_name, LUDB_data=LUDB_data)
    fig, ax = plt.subplots()
    plot_lead_signal_to_ax(signal_mV=signal_mV, ax=ax)

    # Придумываем интервал и рисуем
    delineation_interval = DelineationInterval(t_start=1, t_end=2.1, wave_type=WAVES_TYPES.QRS)
    delineation_interval.draw(ax)
    # показываем итог: сигнал и облако активаций поверх него
    ax.legend()
    plt.show()
