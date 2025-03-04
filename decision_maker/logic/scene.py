class Scene:
    def __init__(self, signals, leads_names):
        self.object_id_generator = -1
        self.signals = signals

        # хранилище всех объектов, каждому соотв. его уникальный object_id
        self.scene_objects_dict = {} # {object_id: object}

        # распределение объектов по отведениям
        self.leads_names_to_object_lists = {} # {lead_name: [object_id1,....]}

        # вначале каждому отведению соотв. пустой список объектов
        for lead_name in leads_names:
            self.leads_names_to_objects[lead_name] = []

    def get_signal_by_lead_name(self, lead_name):
        pass

    def get_leads_names(self):

    def add_delin_point(self, lead_name):
        pass

    def add_delin_interval(self, lead_name, delineation_interval):
        pass

    def add_search_interval(self, lead_name, search_interval):
        pass

    def add_activations(self, lead_name, activations):

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