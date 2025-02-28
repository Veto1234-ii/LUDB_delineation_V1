

from decision_maker.logic.decisions_results import *
from decision_maker.GUI.decisions_results_drawers import *

# Функция для отрисовки объекта "Результат решения"
def draw_decision_result_to_ax(decision_result, ax):
    drawer = DrawerFactory.get_drawer(decision_result)
    drawer.draw(ax)



# Соответствие
class DrawerFactory:
    @staticmethod
    def get_drawer(decision_result):
        if isinstance(decision_result, BinaryActivations):
            return BinaryActivationsDrawer(decision_result)
        elif isinstance(decision_result, SegmentationActivations):
            return SegmentationActivations_Drawer(decision_result)
        elif isinstance(decision_result, DelineationPointsWeak):
            return DelineationPointsWeak_Drawer(decision_result)
        elif isinstance(decision_result, DelineationPointsStrong):
            return DelineationPointsStrong_Drawer(decision_result)
        elif isinstance(decision_result, DelineationIntervalStrong):
            return DelineationIntervalStrong_Drawer(decision_result)
        elif isinstance(decision_result, IntervalSearch):
            return IntervalSearch_Drawer(decision_result)
        else:
            raise ValueError(f"No drawer found for {type(decision_result)}")