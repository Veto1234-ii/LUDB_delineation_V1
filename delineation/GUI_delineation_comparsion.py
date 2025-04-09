from delineation import DelineationOnePoint
from datasets.LUDB_utils import get_LUDB_data, get_test_and_train_ids, get_signal_by_id_and_lead_mV, get_one_lead_delineation_by_patient_id
from settings import LEADS_NAMES, POINTS_TYPES, POINTS_TYPES_COLORS, FREQUENCY
from visualisation_utils.plot_one_lead_signal import plot_lead_signal_to_ax
import matplotlib.pyplot as plt
import numpy as np

class PatientContainer:
    
    def __init__(self, true_delinations, our_delineations, signals_list_mV, leads_names_list, patient_id='-1'):
        """
        сигнал отведений пациента + правильная разметка + наша разметка.
        это три вещи, нужные, чтоб для данного пациента наглядтно посмотреть в GUI, насколько хорошо наша модель его разметила.
        класс является служебным для GUI_DekineationComparsion.
        Args:
            true_delinations: список объектов DelineationOnePoint - правильная разметка. Список может содержать от 1 до 12x9 элементов
            our_delineations: список объектов DelineationOnePoint - наша разметка. Список может содержать от 1 до 12x9 элементов
            signals_list_mV: сигналы нескольких отведений
            leads_names_list: имена этих отведений, взятых из LEADS_NAMES
            patient_id: id пациента в датасете
        """
        # реализация хранения данных пациента: сигналы, разметки и id
        self.true_delinations = true_delinations
        self.our_delineations = our_delineations
        self.signals_list_mV = signals_list_mV
        self.leads_names_list = leads_names_list
        self.patient_id = patient_id

class GUI_DelineationComparison:
    """
    Листалка для сравнения разметки с использованием готовой миллиметровки из plot_one_lead_signal.py
    """
    def __init__(self, patient_containers):
        self.patient_containers = patient_containers
        self.current_patient_index = 0
        self.fig, self.ax = plt.subplots(
            len(self.patient_containers[0].leads_names_list), 
            1, 
            figsize=(15, 10)
        )
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.legend = None  # Будем хранить ссылку на легенду
        self.update_plot()

    def _plot_delineation(self, ax, signal, points, lead_name, is_true_delineation=True):
        """Отрисовка разметки с цветами по типам точек"""
        x_values = np.arange(0, len(signal)) / FREQUENCY

        # Фильтруем точки только для текущего отведения
        points_for_lead = [p for p in points if p.lead_name == lead_name]

        for point in points_for_lead:
            color = POINTS_TYPES_COLORS[point.point_type]
            if is_true_delineation:
                ax.scatter(
                    np.array(point.delin_coords) / FREQUENCY,
                    [signal[int(coord)] for coord in point.delin_coords],
                    color=color,
                    label=f'True {point.point_type}',
                    marker='o',
                    s=20,
                    alpha=0.7,
                    zorder=5
                )
            else:
                for coord in point.delin_coords:
                    ax.axvline(
                        x=coord / FREQUENCY,
                        color=color,
                        linestyle='--',
                        linewidth=1,
                        label=f'Our {point.point_type}',
                        alpha=0.7,
                        zorder=4
                    )

    def update_plot(self):
        patient = self.patient_containers[self.current_patient_index]
        
        # Удаляем предыдущую легенду, если она существует
        if self.legend is not None:
            self.legend.remove()
            self.legend = None
            
        # Собираем все уникальные элементы для легенды
        all_handles = []
        all_labels = []
        
        for i, (lead_name, signal) in enumerate(zip(patient.leads_names_list, patient.signals_list_mV)):
            self.ax[i].clear()
            plot_lead_signal_to_ax(signal_mV=signal, ax=self.ax[i])

            # Передаем текущее имя отведения в метод отрисовки
            self._plot_delineation(self.ax[i], signal, patient.true_delinations, lead_name, is_true_delineation=True)
            self._plot_delineation(self.ax[i], signal, patient.our_delineations, lead_name, is_true_delineation=False)
            
            self.ax[i].set_title(f'Patient {patient.patient_id}, Lead {lead_name}')
            
            # Получаем текущие handles и labels, но не создаем легенду здесь
            handles, labels = self.ax[i].get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)
        
        # Убираем дубликаты в легенде
        unique = dict(zip(all_labels, all_handles))
        
        # Создаем одну общую легенду для всей фигуры
        if unique:  # только если есть что отображать
            self.legend = self.fig.legend(
                unique.values(),
                unique.keys(),
                loc='upper right',
                bbox_to_anchor=(1.0, 1.0),
                borderaxespad=0.5,
                ncol=2,
                fontsize='small',
                framealpha=0.9
            )

        plt.tight_layout()
        plt.draw()

    def on_key_press(self, event):
        # обработка нажатий клавиш для листания пациентов
        if event.key == 'right':
            self.current_patient_index = (self.current_patient_index + 1) % len(self.patient_containers)
            self.update_plot()
        elif event.key == 'left':
            self.current_patient_index = (self.current_patient_index - 1) % len(self.patient_containers)
            self.update_plot()

if __name__ == "__main__":
    # загрузка данных LUDB
    LUDB_data = get_LUDB_data()
    test_patient_ids, _ = get_test_and_train_ids(LUDB_data)

    # выбор первых нескольких пациентов и интересующих отведений и типов точек
    patient_ids = test_patient_ids[0:10]  # первые несколько пациентов
    lead_names = [LEADS_NAMES.i, LEADS_NAMES.ii, LEADS_NAMES.iii]
    points_types = [POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK, POINTS_TYPES.P_PEAK]

    patient_containers = []

    # создание контейнеров для каждого пациента
    for patient_id in patient_ids:
        # получение сигналов интересующих отведений пациента
        signals_list_mV = []
        for lead_name in lead_names:
            signals_list_mV.append(get_signal_by_id_and_lead_mV(patient_id, lead_name=lead_name, LUDB_data=LUDB_data))

        # создание правильной и случайной (нашей) разметки для пациента
        true_delinations = []
        our_delineations = []
        for lead_name in lead_names:
            for point_type in points_types:

                # истинная разметка
                point_delineation = get_one_lead_delineation_by_patient_id(patient_id, LUDB_data, lead_name=lead_name, point_type=point_type)
                point_delineation = [x * FREQUENCY for x in point_delineation]
                true_delineation_obj = DelineationOnePoint(point_type, lead_name, delin_coords=point_delineation, delin_weights=None)
                true_delinations.append(true_delineation_obj)

                # случайная разметка (пример для теста GUI)
                num_points = 4
                point_delineation_random = np.random.randint(0, len(signals_list_mV[0]), num_points)  # случайные координаты
                delin_weights_random = np.random.rand(num_points)  # случайные веса
                our_delineation_obj = DelineationOnePoint(point_type, lead_name, delin_coords=point_delineation_random, delin_weights=delin_weights_random)
                our_delineations.append(our_delineation_obj)

        # создание контейнера для пациента и добавление его в список
        container = PatientContainer(true_delinations, our_delineations, signals_list_mV, lead_names, patient_id=str(patient_id))
        patient_containers.append(container)

    # создание и запуск листалки
    gui = GUI_DelineationComparison(patient_containers=patient_containers)
    plt.show()