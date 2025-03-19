from delineation import DelineationOnePoint
from datasets.LUDB_utils import get_LUDB_data, get_test_and_train_ids, get_signal_by_id_and_lead_mV, get_one_lead_delineation_by_patient_id
from settings import LEADS_NAMES, POINTS_TYPES

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

import matplotlib.pyplot as plt
import numpy as np

class GUI_DekineationComparsion:
    """
    листалка для показа того, как выглядят наша и докторская разметка на разных пациентах.
    разметка берется только по точкам заданных типов в заданных отведениях.
    """
    def __init__(self, patient_containers):

        # инициализация листалки: хранение списка пациентов и текущего индекса
        self.patient_containers = patient_containers
        self.current_patient_index = 0

        # создание графического интерфейса с использованием matplotlib
        self.fig, self.ax = plt.subplots(len(self.patient_containers[0].leads_names_list), 1, figsize=(10, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)  # подключение обработчика клавиш
        self.update_plot()  # обновление графика для первого пациента

    def update_plot(self):

        # получение данных текущего пациента
        patient = self.patient_containers[self.current_patient_index]

        # отображение сигналов и разметки для каждого отведения
        for i, lead_name in enumerate(patient.leads_names_list):
            self.ax[i].clear()
            signal = patient.signals_list_mV[i]
            self.ax[i].plot(signal, label=f'Signal {lead_name}')
            
            # отображение правильной разметки (зеленые точки)
            for point in patient.true_delinations:
                if point.lead_name == lead_name:
                    self.ax[i].scatter(point.delin_coords, [signal[int(coord)] for coord in point.delin_coords], 
                                       color='green', label='True Delineation', zorder=5)
            
            # отображение нашей разметки (красные точки)
            for point in patient.our_delineations:
                if point.lead_name == lead_name:
                    self.ax[i].scatter(point.delin_coords, [signal[int(coord)] for coord in point.delin_coords], 
                                       color='red', label='Our Delineation', zorder=5)
            
            # добавление легенды и заголовка для каждого графика
            self.ax[i].legend()
            self.ax[i].set_title(f'Lead {lead_name}')
        
        # улучшение компоновки графиков
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
    points_types = [POINTS_TYPES.QRS_PEAK, POINTS_TYPES.QRS_END]

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
                true_delineation_obj = DelineationOnePoint(point_type, lead_name, delin_coords=point_delineation, delin_weights=None)
                true_delinations.append(true_delineation_obj)

                # случайная разметка (пример для теста GUI)
                point_delineation_random = np.random.randint(0, len(signals_list_mV[0]), 5)  # случайные координаты
                delin_weights_random = np.random.rand(5)  # случайные веса
                our_delineation_obj = DelineationOnePoint(point_type, lead_name, delin_coords=point_delineation_random, delin_weights=delin_weights_random)
                our_delineations.append(our_delineation_obj)

        # создание контейнера для пациента и добавление его в список
        container = PatientContainer(true_delinations, our_delineations, signals_list_mV, lead_names, patient_id=str(patient_id))
        patient_containers.append(container)

    # создание и запуск листалки
    gui = GUI_DekineationComparsion(patient_containers=patient_containers)
    plt.show()