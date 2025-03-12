from settings import POINTS_TYPES, LEADS_NAMES

class DelineationOnePoint:
    def __init__(self, point_type, lead_name, delin_coords, delin_weights=None):
        """
        Хранит разметку данного типа точки (например, "конец QRS") для сигнала данного отведения.
        delin_coords, delin_weights индексируются одним индексом.
        Args:
            point_type: (POINTS_TYPES) один из 9 типов точек
            lead_name: (LEADS_NAMES) имя отведения, в коротом она ставится
            delin_coords:  (list(int)) координаты, в которых поставились экземпляры этой точки, от 0 до MAX_SIGNAL_LEN-1
            delin_weights: (list(float) or None) каждой координате сопоставляется уверенность от 0 до 1.
            Если массив уверенностей не передали, то назвачем все уверенности единицчными
        """
        self.point_type =point_type
        self.lead_name = lead_name
        self.delin_coords = delin_coords
        self.delin_weights = delin_weights # TODO если None, то забить единицами

    def __len__(self):
        return len(self.delin_coords)

