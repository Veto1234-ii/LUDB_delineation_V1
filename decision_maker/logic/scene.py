from settings import FREQUENCY

class Scene:
    def __init__(self, signals, leads_names):
        self.object_id_generator = -1
        self.signals = signals
        self.leads_names = leads_names

        # переведем время в секунды
        self.t = [i/FREQUENCY for i in range(len(self.signals[0]))]

        # хранилище всех объектов, каждому соотв. его уникальный object_id
        self.scene_objects_dict = {} # {object_id: object}

        # распределение объектов по отведениям
        self.leads_names_to_object_lists = {} # {lead_name: [object_id1,....]}

        # вначале каждому отведению соотв. пустой список объектов
        for lead_name in leads_names:
            self.leads_names_to_objects[lead_name] = []

    def get_signal_by_lead_name(self, lead_name):
        index = self.leads_names.index(lead_name)
        return self.signals[index]

    def get_leads_names(self):
        return self.leads_names

    def add_object(self, lead_name, scene_object):
        self.object_id_generator+=1
        self.leads_names_to_object_lists[lead_name].append(self.object_id_generator)
        self.scene_objects_dict[self.object_id_generator] = scene_object


    def get_nearest_delin_point(self, t, point_type, to_left, lead_name):
        return

    def get_nearest_delin_interval(self, t, wave_type, to_left, lead_name):
        return

    def get_all_points_in_interval(self, t_start, t_end, lead_name):
        return

    def get_all_delin_intervals_in_interval(self, t_start, t_end, lead_name):
        pass

    def delete_activations(self):
        pass

    def delete_oblect_by_id(self, id):
        pass

    def get_all_objects_in_search_interval(self):
        pass

    def draw(self, ax_list):
        pass

    def get_all_undelineated_ts(self):
        pass