from settings import POINTS_TYPES_COLORS, DELINEATION_LINEWIDTH


class DelineationPoint:
    def __init__(self, t, point_type, lead_name, sertainty=1):
        self.t = t
        self.point_type = point_type
        self.sertainty = sertainty
        self.lead_name = lead_name

        self.id_in_scene = None  # Автоматически назначается сценой

    def draw(self, ax, y_max):
        color = POINTS_TYPES_COLORS[self.point_type]
        ax.axvline(x=self.t, color=color, linewidth=DELINEATION_LINEWIDTH, alpha=0.5)

        # Отразим уровень уверенности (максимально возможный и реальный)
        ax.scatter(self.t, y_max, marker='x', alpha=0.5, color='black', s=10)
        ax.scatter(self.t, self.sertainty * y_max, marker='o', color='black', alpha=0.5, s=10)


if __name__ == "__main__":
    from settings import LEADS_NAMES, POINTS_TYPES
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

    # Придумываем точку
    point = DelineationPoint(t=2.3, point_type=POINTS_TYPES.QRS_PEAK, lead_name=LEADS_NAMES.i, sertainty=0.6)
    y_max = max(signal_mV)
    point.draw(ax, y_max=y_max)

    # показываем итог: сигнал и облако активаций поверх него
    ax.legend()
    plt.show()
