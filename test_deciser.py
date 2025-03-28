from decision_maker import Deciser, Scene, DelineationPoint
from datasets import get_test_and_train_ids, get_LUDB_data, get_signals_by_id_several_leads_mkV, \
    get_one_lead_delineation_by_patient_id
from settings import LEADS_NAMES_ORDERED, LEADS_NAMES, POINTS_TYPES, MAX_SIGNAL_LEN
from decision_maker import Deciser, Scene
from delineation import get_F1


class TestReport:
    """
    Объект отчета о тестировани, хранит посчитаннные метрики
    качества (F1, err) нашего алгоритма на тестовом множестве пациента.
    Считает метрики не сам, а заполняется классом MainMetricsTester.
    """
    def __init__(self):
        self.leads_to_points = {}  # { lead_name: {point_type: {'f1':F1, 'err':err}}}

    def _set_F1_err(self, F1, err, point_type, lead_name):
        if lead_name not in self.leads_to_points:
            self.leads_to_points[lead_name]={}
        self.leads_to_points[lead_name][point_type] = {'f1':F1, 'err':err}

    def get_mean_F1_across_all_points(self):
        """ Получить  среднее F1,  дополнительно усредненное
         по всем типам точек по всем отведениям """
        F1_sum = 0
        num_points = 0
        for lead_name, lead_points_info in self.leads_to_points.items():
            for point_type, point_info in lead_points_info.items():
                f1 = point_info['f1']
                F1_sum+=f1
                num_points += 1
        if num_points == 0:
            return -1
        return F1_sum/num_points

    def get_mean_F1_across_points_of_type(self, point_type):
        """ Получить  среднее F1 для данного типа точек (например, пик QRS),
                дополнительно усредненно по всем отведениям,
                где этот тип точек ставился нашим алгоритмом """
        F1_sum = 0
        num_points = 0
        for lead_name, lead_points_info in self.leads_to_points.items():
            for point_type_real, point_info in lead_points_info.items():
                if point_type_real == point_type:
                    f1 = point_info['f1']
                    F1_sum += f1
                    num_points += 1
        if num_points == 0:
            return -1
        return F1_sum / num_points

    def get_mean_abs_err_across_all_points(self):
        """ Получить  среднее отклонение от правильной разметки,
        дополнительно усредненное по всем типам точек по всем отведениям """
        err_sum = 0
        num_points = 0
        for lead_name, lead_points_info in self.leads_to_points.items():
            for point_type, point_info in lead_points_info.items():
                err = point_info['err']
                err_sum += err
                num_points += 1
        if num_points == 0:
            return -1
        return err_sum / num_points

    def get_mean_abs_err_across_points_of_type(self):
        """ Получить  среднее отклонение от правильной разметки,
        для данного типа точек (например, пик QRS),
                дополнительно усредненно по всем отведениям,
                где этот тип точек ставился нашим алгоритмом """
        err_sum = 0
        num_points = 0
        for lead_name, lead_points_info in self.leads_to_points.items():
            for point_type, point_info in lead_points_info.items():
                err = point_info['err']
                err_sum += err
                num_points += 1
        if num_points == 0:
            return -1
        return err_sum / num_points


class _PointStatistics:
    """
    Вспомогательный класс для класса MainMetricsTester (сам по себе не используется)
    """

    def __init__(self, lead_name, point_type, signal_len, tolerance=25):
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
    """
    Класс, который тестирует Deciser по метрицам качества F1, err и записывает результаты тестирования в объект TestReport
    """

    def __init__(self, test_patients_ids, LUDB_data):
        self.test_patients_ids = test_patients_ids

        self.points_we_want = {LEADS_NAMES.i: [POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK],
                               LEADS_NAMES.ii: [], #[POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK],
                               LEADS_NAMES.iii: [] #POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK,
                                                 #POINTS_TYPES.T_PEAK]
        }  # TODO из Deciser, если Катя написала в нем соотв. метод

        self.LUDB_data = LUDB_data

        # Основная структура данных - список экземпляров класса PointStatistics
        self.points_statistics_list = []
        for lead_name in self.points_we_want.keys():
            for point_type in self.points_we_want[lead_name]:
                self.points_statistics_list.append(_PointStatistics(lead_name, point_type, MAX_SIGNAL_LEN))

    def _get_entry_index(self, lead_name, point_type):
        for i in range(len(self.points_statistics_list)):
            if self.points_statistics_list[i].point_type == point_type:
                if self.points_statistics_list[i].lead_name == lead_name:
                    return i
        return None

    def _register_scene_to_statistics(self, scene, patient_id):
        for lead_name in self.points_we_want.keys():
            for point_type in self.points_we_want[lead_name]:
                # из сцены вытаскиваем нашу разметку этой точки в этом отведении
                our_coords = scene.get_binary_delineation_by_point_type_and_lead(lead_name,
                                                                                 point_type)  # dвремя не в секундах

                # вытаскиваем верную разметку этой точки в этом отведении
                true_coords = get_one_lead_delineation_by_patient_id(patient_id=patient_id,
                                                                     LUDB_data=self.LUDB_data,
                                                                     lead_name=lead_name,
                                                                     point_type=point_type)

                index_of_stat_entry = self._get_entry_index(lead_name, point_type)
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
            deciser = Deciser()

            # генерируем разметку нашим алгоритмом - получаем заполненную сцену для данного пациента
            scene, _ = deciser.run(leads_names=leads_names, signals=signals_list_mkV)
            self._register_scene_to_statistics(scene=scene, patient_id=patient_id)

    def _statistics_to_report(self):
        report = TestReport()
        for point_statistiscs in self.points_statistics_list:
            F1, err = point_statistiscs.get_F1_err()
            report._set_F1_err(F1=F1,
                               err=err,
                               point_type=point_statistiscs.point_type,
                               lead_name=point_statistiscs.lead_name)
        return report


    def run(self):
        self._fill_statistics()
        report = self._statistics_to_report()
        return report


if __name__ == "__main__":
    from datasets import get_test_and_train_ids, get_LUDB_data

    LUDB_data = get_LUDB_data()
    test_patients_ids, _ = get_test_and_train_ids(LUDB_data)

    tester = MainMetricsTester(test_patients_ids[0:2], LUDB_data)

    # Получаем объект отчета о тестировани, содержаший посчитаннные метрики
    # качества нашего алгоритма на данном тестовом множестве пациента:
    report = tester.run()

    # Распечатываем из него важные нам вещи:
    # Средние значения метрик по всем видам точек во всех отведениях
    F1 = report.get_mean_F1_across_all_points()
    print("По всем видам точек: F1 "+ str(F1))

