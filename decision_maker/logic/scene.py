from settings import FREQUENCY
from scene_objects import DelineationPoint, DelineationInterval, SearchInterval, Activations


class Scene:
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
        delin_points = []
        for id, obj in self.scene_objects_dict.items():
            if isinstance(obj, DelineationPoint):
                if lead_name == obj.lead_name:
                    if point_type is None or obj.point_type == point_type:
                        delin_points.append(obj)

        sorted_from_min_to_max = sorted(delin_points, key=lambda obj: abs(obj.t))
        return sorted_from_min_to_max


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
                if first_id == point_id
                    return obj
        return None

    def get_intervals_by_end_point(self, point_id):
        for _, obj in self.scene_objects_dict.items():
            if isinstance(obj, DelineationInterval):
                end_id = obj.delin_point_start.id
                if end_id == point_id
                    return obj
        return None


    def draw(self, ax_list, leads_names, ids, y_max):
        for id in ids:
            drawable_obj = self.scene_objects_dict[id]
            lead_name = drawable_obj.lead_name
            index_of_ax = leads_names.index(lead_name)
            drawable_obj.draw(ax=ax_list[index_of_ax], y_max=y_max)

