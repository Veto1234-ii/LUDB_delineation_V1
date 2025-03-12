from delineation import DelineationOnePoint
from datasets.LUDB_utils import get_LUDB_data, get_test_and_train_ids, get_signal_by_id_and_lead_mV, get_one_lead_delineation_by_patient_id
from settings import LEADS_NAMES, POINTS_TYPES

class PatientContainer:
    def __init__(self, true_delinations, our_delineations, signals_list_mV, leads_names_list, patient_id='-1'):
        pass

class GUI_DekineationComparsion:
    """
    Листалка для показа того, как выглядят наша и докторсккая разметка на разных пациентах.
    Разметка берется только по точкам заданных типов в заданных отведениях.
    """
    def __init__(self, patient_containers):
        pass


if __name__ == "__main__":
    LUDB_data = get_LUDB_data()
    test_patient_ids, _ = get_test_and_train_ids(LUDB_data)

    patient_ids = test_patient_ids[0:10] #первые несколько т
    lead_names = [LEADS_NAMES.i, LEADS_NAMES.ii, LEADS_NAMES.iii ]
    points_types = [POINTS_TYPES.QRS_PEAK, POINTS_TYPES.QRS_END]

    patient_containers  = []

    for patient_id in patient_ids:
        # получаем сигналы интересующих нас отведений пациента
        signals_list_mV = []
        for lead_name in lead_names:
            signals_list_mV.append(get_signal_by_id_and_lead_mV(patient_id, lead_name=lead_name, LUDB_data=LUDB_data))

        # истинная и наша (сгенерирована для примера случайно) разметка для этого пациента
        true_delinations = []
        our_delineations = []
        for lead_name in lead_names:
            for point_type in points_types:
                # истиннная разметка этой точки в этом отведении - заворачиваем в DelineationOnePoint
                point_delineation = get_one_lead_delineation_by_patient_id(patient_id, LUDB_data, lead_name=lead_name, point_type=point_type)
                true_delineation_obj = DelineationOnePoint(point_type, lead_name, delin_coords=point_delineation, delin_weights=None)
                true_delinations.append(true_delineation_obj)

                # наша разметка этой точки в этом отведении - заворачиваем в DelineationOnePoint (тут сгенерируем ее случайно, т.к. это просто пример для теста GUI)
                point_delineation_random = 5 случайныйх int координат, каждая в промежутке от 0 до MAX_SIGNAL_LEN-1
                delin_weights_random = 5 случайных float числел, каждое в промежутке от 0 до 1
                our_delineation_obj = DelineationOnePoint(point_type, lead_name, delin_coords=point_delineation_random, delin_weights=delin_weights_random)
                our_delineations.append(our_delineation_obj)

        # Все данные по этому пациенту собраны (его сигнал и разметки нескольких видов точек). Положим этого пациента к оастальным пациентам.
        container = PatientContainer(true_delinations, our_delineations, signals_list=signals_list_mV, leads_names_list=lead_names, patient_id=str(patient_id))
        patient_containers.append(container)

        # Создаем листалку
        gui = GUI_DekineationComparsion(patient_containers=patient_containers)