from .drawing_scenario import Scenario
from draw_decision_result import draw_decision_result_to_ax

class GUI_MainForm:
    def __init__(self, signals, leads_names, drawing_scenario):
        self.i = 0
        # TODO создаем GUI - форму с матлотлиб fig для отрисовки отведений и текстовое окошко для лога
        # Рисуем в картинку сигналы
        # привязываем self.on_next_step в кач-ве обработчика к событию "клик по стрелочке вправо на клаивтуре"
        # а также и self.on_prev_step
        pass



    def on_next_step(self):
        # TODO очищаем всю картинку и лог
        # TODO рисуем заново сигнал
        decisions_results = self.scenario.get_next_group_to_draw()
        if decisions_results is None:
            print(" Распозавание завершено")
            return
        self._draw_decisions_group_to_GUI(decisions_results)



    def _draw_decisions_group_to_GUI(self, decisions_results):
        for decision_result in decisions_results:
            ax = 0  # TODO взяв имя отведения scenario_entry.lead_name, находим соотв. ему ax
            draw_decision_result_to_ax(decision_result, ax=ax)
            # логируем
