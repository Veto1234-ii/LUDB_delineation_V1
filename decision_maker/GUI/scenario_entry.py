from settings import LEADS_NAMES
from decision_maker.decision_maker_settings import DECISION_MAKER_COLORS

class ScenarioEntry:
    def __init__(self, color_name=DECISION_MAKER_COLORS.QRS_PEAK, coords_t=[1,2,4], lead_name=LEADS_NAMES.i, legend="", text_to_log="", Y_MAX_mV=0.3):
        self.color_name = color_name
        self.coords_t = coords_t
        self.lead_name = lead_name
        self.legend = legend
        self.Y_MAX_mV = Y_MAX_mV

        self.text_to_log = text_to_log


    def draw_to_ax(self, ax):
        # TODO вызываем метод plot_one_lead_delineation из visualisation_utils
        # текст для лога в рисовании не участвует
        pass