from settings import WAVES_TYPES_COLORS, WAVES_TYPES, POINTS_TYPES

class DelineationInterval:
    def __init__(self, delin_point_start, delin_point_end):
        """
        Объект сцены, инкапсулирующий интервал разметки, у интервала определенный тип, начало и конец
        Args:
            delin_point_start: объект = точка разметки (старт интервала)
            delin_point_end: объект = точка разметки  (конец интервала )


        """
        self.delin_point_start= delin_point_start
        self.delin_point_end = delin_point_end

        if delin_point_start.point_type == POINTS_TYPES.QRS_START:
            self.interval_type = WAVES_TYPES.QRS
        else:
            if delin_point_start.point_type == POINTS_TYPES.T_START:
                self.interval_type = WAVES_TYPES.T
            else:
                if delin_point_start.point_type == POINTS_TYPES.P_START:
                    self.interval_type = WAVES_TYPES.P
                else:
                    self.interval_type = WAVES_TYPES.NO_WAVE

        self.lead_name = delin_point_start.lead_name

        self.id_in_scene = None  # Автоматически назначается сценой

    def contain(self, t):
        t_start = self.delin_point_start.t
        t_end = self.delin_point_end.t
        return t<= t_end and t>= t_start

    def draw(self, ax):
        color = WAVES_TYPES_COLORS[self.interval_type]
        t_start = self.delin_point_start.t
        t_end = self.delin_point_end.t
        ax.axvspan(t_start, t_end, color=color, alpha=0.3)


if __name__ == "__main__":
    from settings import LEADS_NAMES, FREQUENCY, WAVES_TYPES, POINTS_TYPES
    from datasets.LUDB_utils import get_some_test_patient_id, get_signal_by_id_and_lead_mV, get_LUDB_data
    from visualisation_utils import plot_lead_signal_to_ax
    from delineation_point import DelineationPoint

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
    point1 = DelineationPoint(t=1, point_type=POINTS_TYPES.QRS_START, lead_name=LEADS_NAMES.i, sertainty=0.6)
    point2 = DelineationPoint(t=2.1, point_type=POINTS_TYPES.QRS_END, lead_name=LEADS_NAMES.i, sertainty=0.6)
    delineation_interval = DelineationInterval(delin_point_start=point1, delin_point_end=point2)
    delineation_interval.draw(ax)

    plt.show()
