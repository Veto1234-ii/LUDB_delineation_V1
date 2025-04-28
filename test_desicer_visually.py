from delineation import DelineationOnePoint
from datasets.LUDB_utils import get_LUDB_data, get_test_and_train_ids, get_signal_by_id_and_lead_mV, get_one_lead_delineation_by_patient_id, get_signals_by_id_several_leads_mkV
from settings import LEADS_NAMES, POINTS_TYPES, POINTS_TYPES_COLORS, FREQUENCY
from visualisation_utils.plot_one_lead_signal import plot_lead_signal_to_ax
from delineation import GUI_DelineationComparison, PatientContainer, DelineationOnePoint
from decision_maker import Deciser, Deciser_leads

import matplotlib.pyplot as plt
import numpy as np
import sys
import io


def get_delinations_objects_from_scene(scene, patient_id, LUDB_data, points_we_want):
    true_delinations = []  # список объектов DelineationOnePoint
    our_delineations = []

    for lead_name in points_we_want.keys():
        for point_type in points_we_want[lead_name]:
            # из сцены вытаскиваем нашу разметку этой точки в этом отведении
            our_coords = scene.get_binary_delineation_by_point_type_and_lead(lead_name,
                                                                             point_type)  # время не в секундах
            our_delineation_obj = DelineationOnePoint(point_type, lead_name, delin_coords=our_coords)
            our_delineations.append(our_delineation_obj)

            # вытаскиваем верную разметку этой точки в этом отведении
            true_coords_in_seconds = get_one_lead_delineation_by_patient_id(patient_id=patient_id,
                                                                            LUDB_data=LUDB_data,
                                                                            lead_name=lead_name,
                                                                            point_type=point_type)
            true_coords = [true_coords_in_seconds[i] * FREQUENCY for i in range(len(true_coords_in_seconds))] # хотим и это не в секундах

            true_delineation_obj = DelineationOnePoint(point_type, lead_name, delin_coords=true_coords,
                                                       delin_weights=None)
            true_delinations.append(true_delineation_obj)
    return our_delineations, true_delinations



if __name__ == "__main__":
    # загрузка данных LUDB
    LUDB_data = get_LUDB_data()

    # на каких пациентах смотрим работу алгоритма
    test_patient_ids_all, _ = get_test_and_train_ids(LUDB_data)
    test_patient_ids = test_patient_ids_all[3]

    # экземпляр алгоритма
    deciser = Deciser_leads()

    # сигнал каких отведений показывать
    leads_names = [LEADS_NAMES.i, LEADS_NAMES.ii, LEADS_NAMES.iii]

    # в какие отведения и какие типы точек умеет ставить алгоритм
    points_we_want = deciser.what_points_we_want()

    # создание контейнеров для каждого пациента
    patient_containers = []
    for patient_id in test_patient_ids:

        # для всех интересующих нас отведений пациента вытаскиваем сигнал (нам для разных использований потреуется и в микро-, и в милливольтах)
        signals_list_mV = []  # в милливольтах
        for lead_name in leads_names:
            lead_signal = get_signal_by_id_and_lead_mV(patient_id, lead_name, LUDB_data)
            signals_list_mV.append(lead_signal)

        signals_list_mkV, leads_names_list_mV = get_signals_by_id_several_leads_mkV(patient_id, LUDB_data, leads_names) # в микровольтах

        # Подавление вывода (чтобы deciser не мусорил в конксоль всякой отладочной информацией)
        original_stdout = sys.stdout  # Сохраняем оригинальный stdout
        sys.stdout = io.StringIO()  # Перенаправляем вывод в "никуда"

        
        # генерируем разметку нашим алгоритмом - получаем заполненную сцену для данного пациента
        scene, history = deciser.run(leads_names=leads_names, signals=signals_list_mkV)
        

        sys.stdout = original_stdout  # Восстанавливаем stdout

        # результаты распознаваия хранятся в сцене, теперь созданим на их основе объект PatientContainer
        our_delineations, true_delinations = get_delinations_objects_from_scene(scene, patient_id, LUDB_data, points_we_want)
        container = PatientContainer(true_delinations, our_delineations, signals_list_mV, leads_names,
                                         patient_id=str(patient_id))
        patient_containers.append(container)

        # чтобы память не переполнялась, явно удалим все что можно
        del scene
        del history
        deciser.clear_scene()

    # создание и запуск листалки
    gui = GUI_DelineationComparison(patient_containers=patient_containers)
    plt.show()