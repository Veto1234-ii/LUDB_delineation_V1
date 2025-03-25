from operator import index

from decision_maker import Deciser, Scene, DelineationPoint
from datasets import get_test_and_train_ids, get_LUDB_data, get_signals_by_id_several_leads_mkV, get_one_lead_delineation_by_patient_id
from settings import LEADS_NAMES_ORDERED, LEADS_NAMES, POINTS_TYPES, MAX_SIGNAL_LEN
from decision_maker import Deciser, Scene
from delineation import get_F1

class TestReport:
    def __init__(self):
        """
        Хранит итоговый отчет тестирования Deciser-а.
        Для интересующих нас точек хранятся метрики качества их расстановки нашим алгоритмом.

        Args:
            points_types_leads: список кортежей (тип точки, имя отведения)
        """

class PointStatistics:

    def __init__(self, lead_name, point_type,  signal_len, tolerance = 25):
        self.lead_name = lead_name
        self.point_type = point_type
        self.our_delineations = []
        self.true_delineations = []
        self.patient_ids = []
        self.signal_len = signal_len
        self.tolerance = tolerance


    def get_F1_err(self):
        F1, err = get_F1(true_delinations=self.true_delineations,
                         our_delinations=self.our_delineations,
                         tolerance=self.tolerance,
                         len_signal=MAX_SIGNAL_LEN)
        return F1, err


    def add_patient_entry(self, patient_id, our_delineation, true_delineation):
        self.true_delineations.append(true_delineation)
        self.patient_ids.append(patient_id)
        self.our_delineations.append(our_delineation)




class MainMetricsTester:
    def __init__(self, test_patients_ids, LUDB_data):
        self.test_patients_ids = test_patients_ids

        self.points_we_want = {LEADS_NAMES.i:[POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK],
                  LEADS_NAMES.ii:[POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK],
                  LEADS_NAMES.iii:[POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK]}

        self.LUDB_data = LUDB_data

        # Основная структура данных - список экземпляров класса PointStatistics
        self.points_statistics_list = []
        for lead_name in self.points_we_want.keys():
            for point_type in self.points_we_want[lead_name]:
                self.points_statistics_list.append(PointStatistics(lead_name, point_type, MAX_SIGNAL_LEN))

    def get_entry_index(self, lead_name, point_type):
        #TODO
        return index

    def _register_scene_to_statistics(self, scene, patient_id):
        for lead_name in self.points_we_want.keys():
            for point_type in self.points_we_want[lead_name]:
                # из сцены вытаскиваем нашу разметку этой точки в этом отведении
                our_coords = scene.get_binary_delineation_by_point_type_and_lead(lead_name, point_type) # dвремя не в секундах

                # вытаскиваем верную разметку этой точки в этом отведении
                true_coords = get_one_lead_delineation_by_patient_id(patient_id=patient_id,
                                                                     LUDB_data=self.LUDB_data,
                                                                     lead_name=lead_name,
                                                                     point_type=point_type)


                index_of_stat_entry = self.get_entry_index(lead_name, point_type)
                self.points_statistics_list[index_of_stat_entry].add_patient_entry(patient_id=patient_id,
                                                             our_delineation=our_coords,
                                                             true_delineation=true_coords)

    def _fill_statistics(self):
        # какие отведения нас интересуют
        leads_names = list(self.points_we_want.keys())

        for patient_id in self.test_patients_ids:
            # для всех интересующих нас отведений пациента выскиваем сигнал
            signals_list_mkV, _ = get_signals_by_id_several_leads_mkV(patient_id=patient_id,
                                                                      LUDB_data=LUDB_data,
                                                                      leads_names_list=leads_names)
            # создаем экземпляр нашего алгоритма разметки
            deciser = Deciser(leads_names=leads_names, signals=signals_list_mkV)

            # генерируем разметку нашим алгоритмом - получаем заполненную сцену для данного пациента
            scene, _ = deciser.run()
            self._register_scene_to_statistics(scene=scene, patient_id=patient_id)

    def _statistics_to_report(self):
        pass

    def run(self):
        self._fill_statistics()
        report = self._statistics_to_report()
        return report


if __name__ == "__main__":
    from datasets import get_test_and_train_ids, get_LUDB_data

    LUDB_data = get_LUDB_data()
    test_patients_ids, _ = get_test_and_train_ids(LUDB_data)

    tester = MainMetricsTester(test_patients_ids, LUDB_data)

    report = tester.run()

