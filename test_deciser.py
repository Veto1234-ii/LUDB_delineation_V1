from decision_maker import Deciser, Scene, DelineationPoint
from datasets import get_test_and_train_ids, get_LUDB_data, get_signals_by_id_several_leads_mkV, \
    get_one_lead_delineation_by_patient_id
from settings import LEADS_NAMES_ORDERED, LEADS_NAMES, POINTS_TYPES, MAX_SIGNAL_LEN, FREQUENCY, TOLERANCE, POINTS_TYPES_STR_NAMES
from decision_maker import Deciser, Scene
from delineation import get_F1

import sys
import io

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

    def get_mean_abs_err_across_points_of_type(self, point_type):
        """ Получить  среднее отклонение от правильной разметки,
        для данного типа точек (например, пик QRS),
                дополнительно усредненно по всем отведениям,
                где этот тип точек ставился нашим алгоритмом """
        err_sum = 0
        num_points = 0
        for lead_name, lead_points_info in self.leads_to_points.items():
            for point_type_real, point_info in lead_points_info.items():
                if point_type_real == point_type:
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

    def __init__(self, lead_name, point_type, signal_len):
        self.lead_name = lead_name
        self.point_type = point_type
        self.our_delineations = []
        self.true_delineations = []
        self.patient_ids = []
        self.signal_len = signal_len


    def get_F1_err(self):
        F1, err = get_F1(true_delinations=self.true_delineations,
                         our_delinations=self.our_delineations,
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

    def __init__(self, test_patients_ids, LUDB_data, deciser, leads_names):
        self.leads_names = leads_names
        self.test_patients_ids = test_patients_ids
        self.deciser = deciser
        print (f"Кол-во тестовых пациентов {len(self.test_patients_ids)}")

        self.points_we_want = self.deciser.what_points_we_want()  # TODO из Deciser, если Катя написала в нем соотв. метод

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
        log_str = f"{patient_id} "

        for lead_name in self.points_we_want.keys():
            for point_type in self.points_we_want[lead_name]:
                # из сцены вытаскиваем нашу разметку этой точки в этом отведении
                our_coords = scene.get_binary_delineation_by_point_type_and_lead(lead_name,
                                                                                 point_type)  # dвремя не в секундах

                # вытаскиваем верную разметку этой точки в этом отведении
                true_coords_in_seconds = get_one_lead_delineation_by_patient_id(patient_id=patient_id,
                                                                     LUDB_data=self.LUDB_data,
                                                                     lead_name=lead_name,
                                                                     point_type=point_type)
                true_coords = [true_coords_in_seconds[i]* FREQUENCY for i in range(len(true_coords_in_seconds))]

                index_of_stat_entry = self._get_entry_index(lead_name, point_type)
                self.points_statistics_list[index_of_stat_entry].add_patient_entry(patient_id=patient_id,
                                                                                   our_delineation=our_coords,
                                                                                   true_delineation=true_coords)
                F1, err = get_F1(true_delinations=[true_coords],
                         our_delinations=[our_coords],
                         len_signal=MAX_SIGNAL_LEN)
                if err is None:
                    err = -1
                log_str+=f"                 {POINTS_TYPES_STR_NAMES[point_type]}: F1= {F1:.2f}, err ={err:.2f}   "

        print(log_str)

    def _fill_statistics(self):
        # какие отведения нас интересуют



        for patient_id in self.test_patients_ids:
            # для всех интересующих нас отведений пациента выскиваем сигнал
            signals_list_mkV, _ = get_signals_by_id_several_leads_mkV(patient_id=patient_id,
                                                                      LUDB_data=LUDB_data,
                                                                      leads_names_list=self.leads_names)

            # Подавление вывода
            original_stdout = sys.stdout  # Сохраняем оригинальный stdout
            sys.stdout = io.StringIO()  # Перенаправляем в "никуда"

            # генерируем разметку нашим алгоритмом - получаем заполненную сцену для данного пациента
            scene, history = self.deciser.run(leads_names=self.leads_names, signals=signals_list_mkV)
            
            
            sys.stdout = original_stdout  # Восстанавливаем stdout

            self._register_scene_to_statistics(scene=scene, patient_id=patient_id)
            
            del scene
            del history
            self.deciser.clear_scene()

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
    test_patients_ids, train_patients_ids = get_test_and_train_ids(LUDB_data)
    deciser = Deciser()
    leads_names = [LEADS_NAMES.i, LEADS_NAMES.ii, LEADS_NAMES.iii]
    tester = MainMetricsTester(train_patients_ids, LUDB_data, deciser=deciser, leads_names=leads_names)

    # Получаем объект отчета о тестировани, содержаший посчитаннные метрики
    # качества нашего алгоритма на данном тестовом множестве пациента:
    report = tester.run()

    # Распечатываем из него важные нам вещи:

    # Средние значения метрик по всем видам точек во всех отведениях
    F1 = report.get_mean_F1_across_all_points()
    err = report.get_mean_abs_err_across_all_points()
    print(f"              среднее по всем видам точек:  F1 = {F1:.2f} , err = {err:.2f}")

    # Средние значения метрик по всем видам точек поотдельности (но с устреднением по отведениям)
    p1 = report.get_mean_F1_across_points_of_type(point_type=POINTS_TYPES.P_PEAK)
    p2 = report.get_mean_F1_across_points_of_type(point_type=POINTS_TYPES.QRS_PEAK)
    p3 = report.get_mean_F1_across_points_of_type(point_type=POINTS_TYPES.T_PEAK)

    print(f"              F1: пик P {p1:.2f}, пик QRS { p2:.2f}, пик T {p3:.2f}")

    e1 = report.get_mean_abs_err_across_points_of_type(point_type=POINTS_TYPES.P_PEAK)
    e2 = report.get_mean_abs_err_across_points_of_type(point_type=POINTS_TYPES.QRS_PEAK)
    e3 = report.get_mean_abs_err_across_points_of_type(point_type=POINTS_TYPES.T_PEAK)

    print(f"             Подробнее err: пик Р {e1:.2f}, пик QRS {e2:.2f}, пик T {e3:.2f}")

