from decision_maker import Deciser, Scene, DelineationPoint
from datasets import get_test_and_train_ids, get_LUDB_data
from settings import LEADS_NAMES_ORDERED, LEADS_NAMES, POINTS_TYPES

class TestReport:
    def __init__(self, points_types_leads):
        """
        Хранит итоговый отчет тестирования Deciser-а.
        Для интересующих нас точек хранятся метрики качества их расстановки нашим алгоритмом.

        Args:
            points_types_leads: список кортежей (тип точки, имя отведения)
        """

    def add_F1_and_err(self, point_type, point_lead, F1, err):
        pass

    def get_mean_F1_by_point_type(self, point_type):
        pass

    def get_mean_err_by_point_type(self):
        pass

    def __str__(self):
        pass


class MainMetricsTester:
    def __init__(self, deciser, test_patients_ids = None):

        self.deciser = deciser
        self.LUDB_data = get_LUDB_data()

        # Пациенты для тестирования:
        if test_patients_ids is None:
            self.test_patients_ids, _ = get_test_and_train_ids(self.LUDB_data)
        else:
            self.test_patients_ids = test_patients_ids

        self.scenes = self.get_all_scenes()

         = self.deciser.what_points_we_want() # пока хардкодно - пики трех волн первых трех отведений.

        s


    def get_all_scenes(self):
        scenes = []
        for patient_id in self.test_patients_ids:
            scene

    def run():
        report = TestReport()
        return report


if __name__ == "__main__":
    from datasets import get_test_and_train_ids, get_LUDB_data
    deciser = ...# TODO Катя

    LUDB_data = get_LUDB_data()
    test_patients_ids, _ = get_test_and_train_ids(LUDB_data)

    tester = MainMetricsTester(deciser, test_patients_ids)

    report = tester.run()
    print(str(report))
