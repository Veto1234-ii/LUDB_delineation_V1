from settings import POINTS, POINTS_TYPES_COLORS
import matplotlib.pyplot as plt


class BinaryActivations:
    def __init__(self, net_activations, activations_t, type_of_point=POINTS.QRS_PEAK, label=""):
        """

        Args:
            net_activations: список активаций нейросети
            activations_t: координаты (времена) этих активаций, в секундах
            type_of_point: на каком типе точек специализируется эта сеть
            label: строчка c какисм-то пояснением, которая пойдет в лог
        """
        self.net_activations = net_activations
        self.activations_t = activations_t
        self.label = label
        self.type_of_point = type_of_point


    def draw(self, ax, y_max):
        color = POINTS_TYPES_COLORS[self.type_of_point]
        normed_activations = [self.net_activations[i]/y_max for i in range(len(self.net_activations))]
        ax.plot(self.activations_t, normed_activations, color=color, label=self.label, alpha=0.5)



if __name__ == "__main__":
    from settings import LEADS_NAMES,FREQUENCY
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

    # Генерируем случайную серию "активаций"
    t=[i/FREQUENCY for i in range(0, len(signal_mV))]
    y_max = max(signal_mV)
    activations = [abs(math.sin(i)) for i in t]

    # Рисуем ее:
    binary_activations=BinaryActivations(net_activations=activations,
                                         activations_t=t, type_of_point=POINTS.QRS_PEAK, label="тестовые активации")
    binary_activations.draw(ax, y_max=y_max)
    ax.legend()
    plt.show()
