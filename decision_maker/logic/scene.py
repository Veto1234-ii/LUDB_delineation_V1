

from settings import FREQUENCY, LEADS_NAMES, POINTS_TYPES, WAVES_TYPES
from decision_maker.logic import DelineationPoint, DelineationInterval, SearchInterval, Activations

from settings import LEADS_NAMES, FREQUENCY
from datasets.LUDB_utils import get_some_test_patient_id, get_signal_by_id_and_lead_mV, get_LUDB_data
from visualisation_utils import plot_lead_signal_to_ax


import matplotlib.pyplot as plt
from math import sin

class Scene:
    """
    Класс-хранилище для объектов разметки (точек и интервалов), расставляемых нашим основным алгоритмом разметки.
    Типы объектов см. в пакете decision_maker.logic.scene_objects
    """
    def __init__(self):
        self.object_id_generator = -1

        # хранилище всех объектов, каждому соотв. его уникальный object_id
        self.scene_objects_dict = {} # {object_id: object}

    def add_object(self, scene_object):
        self.object_id_generator+=1
        self.scene_objects_dict[self.object_id_generator] = scene_object
        scene_object.id = self.object_id_generator
        return scene_object.id

    def get_all_points_in_lead_sorted(self,  lead_name, point_type=None):
        """
        Получить все объекты класса DelineationPoint для данного типа точек в данном отведении
        Args:
            lead_name: имя отведения
            point_type: тип точки или None (если None, то возвращаем точки всех видов из данного отведения)

        Returns:
            Список объектов типа DelineationPoint
        """
        delin_points = []
        for id, obj in self.scene_objects_dict.items():
            if isinstance(obj, DelineationPoint):
                if lead_name == obj.lead_name:
                    if point_type is None or obj.point_type == point_type:
                        delin_points.append(obj)

        sorted_from_min_to_max = sorted(delin_points, key=lambda obj: abs(obj.t))
        return sorted_from_min_to_max

    def get_binary_delineation_by_point_type_and_lead(self, lead_name, point_type):
        """
        Получить координаты точки данного типа в данном отведении
        Args:
            lead_name: название отведения
            point_type: тип точки

        Returns:
            coords: список чисел, кажждое от 0 до 5000 (т.е. не в секунадах)
        """
        points_objects = self.get_all_points_in_lead_sorted(lead_name=lead_name, point_type=point_type)
        coords = []
        for point_object in points_objects:
            coords.append(point_object.t*FREQUENCY)
        return coords

    def get_nearest_delin_point(self, t, point_type,  lead_name, to_left=True):
        points_in_lead = self.get_all_points_in_lead_sorted(lead_name=lead_name, point_type=point_type)
        if len(points_in_lead)==0:
            return None
        if to_left:
            filtered_objects = [obj for obj in points_in_lead if obj.t < t]
        else:
            filtered_objects = [obj for obj in points_in_lead if obj.t > t]
        nearest = min(filtered_objects, key=lambda obj: abs(obj.t - t))
        return nearest

    def get_all_points_in_interval(self, t_start, t_end, lead_name, point_type):
        points_in_lead = self.get_all_points_in_lead_sorted(lead_name=lead_name, point_type=point_type)
        if len(points_in_lead) == 0:
            return None
        points_in_interval = [obj for obj in points_in_lead if obj.t <= t_end and obj.t>= t_start]
        return points_in_interval

    def get_intervals_by_first_point(self, point_id):
        for _, obj in self.scene_objects_dict.items():
            if isinstance(obj, DelineationInterval):
                first_id = obj.delin_point_start.id
                if first_id == point_id:
                    return obj
        return None

    def get_intervals_by_end_point(self, point_id):
        for _, obj in self.scene_objects_dict.items():
            if isinstance(obj, DelineationInterval):
                end_id = obj.delin_point_start.id
                if end_id == point_id:
                    return obj
        return None

    def get_all_objects_ids(self):
        return list(self.scene_objects_dict.keys())

    def draw(self, ax_list, leads_names, ids, y_max):
        for id in ids:
            drawable_obj = self.scene_objects_dict[id]
            lead_name = drawable_obj.lead_name
            index_of_ax = leads_names.index(lead_name)
            if (isinstance(drawable_obj, DelineationPoint) or
                    isinstance(drawable_obj, Activations)):
                drawable_obj.draw(ax=ax_list[index_of_ax], y_max=y_max)
            else:
                drawable_obj.draw(ax=ax_list[index_of_ax])

def create_test_scene_and_history():
    # Созададим и заполним объектами сцену
    from decision_maker.logic import SceneHistory
    scene = Scene()
    point1 = DelineationPoint(t=1.2,

                              lead_name=LEADS_NAMES.i,
                              point_type=POINTS_TYPES.QRS_START,
                              sertainty=0.8)

    point2 = DelineationPoint(t=2.2,
                              lead_name=LEADS_NAMES.i,
                              point_type=POINTS_TYPES.QRS_END,
                              sertainty=0.5)

    interval = DelineationInterval(delin_point_start=point1, delin_point_end=point2)


    activation_t = [i/FREQUENCY for i in range(1000, 1500)]
    net_activations =[sin(activation) for activation in activation_t]
    activ = Activations(net_activations=net_activations,
                        activations_t=activation_t,
                        color='red',
                        lead_name=LEADS_NAMES.ii)

    search_interval = SearchInterval(t_start=3, t_end=3.7, lead_name=LEADS_NAMES.iii)

    id1 = scene.add_object(point1)
    id2 = scene.add_object(point2)
    id3 = scene.add_object(interval)
    id4 = scene.add_object(activ)
    id5 = scene.add_object(search_interval)

    # Имея id-шники объектов сцены создадим какоую-нибудь историю
    scene_history = SceneHistory()
    scene_history.add_entry(visibles=[id1])
    scene_history.add_entry(invisibles=[id1], visibles=[id2])
    scene_history.add_entry(visibles=[id1, id2, id3])
    scene_history.add_entry(visibles=[id4])
    scene_history.add_entry(visibles=[id5], invisibles=[id3])

    return scene, scene_history



if __name__ == "__main__":
    # Отрисуем поверх сигнала случайного пациента
    LUDB_data = get_LUDB_data()
    patient_id = get_some_test_patient_id()
    lead_name = LEADS_NAMES.i
    signal_mV = get_signal_by_id_and_lead_mV(patient_id, lead_name=lead_name, LUDB_data=LUDB_data)
    fig, ax = plt.subplots()
    plot_lead_signal_to_ax(signal_mV=signal_mV, ax=ax)

    scene, _ = create_test_scene_and_history()
    scene.draw(ax_list=[ax],
               leads_names=[LEADS_NAMES.i],
               y_max=max(signal_mV),
               ids=scene.get_all_objects_ids())

    plt.show()




