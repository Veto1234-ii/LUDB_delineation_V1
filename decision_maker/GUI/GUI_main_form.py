from .drawing_scenario import Scenario
from .scenario_entry import  ScenarioEntry

class GUI_ToShowScenario:
    def __init__(self, signals, leads_names, scenario):
        self.i = 0
        # TODO создаем GUI - форму с матлотлиб fig для отрисовки отведений и текстовое окошко для лога
        # Рисуем в картинку сигналы
        # привязываем self.on_next_step в кач-ве обработчика к событию "клик по стрелочке вправо на клаивтуре"
        pass


    def draw_entry_in_GUI(self, scenario_entry):
        ax= 0 #TODO взяв имя отведения scenario_entry.lead_name, находим соотв. ему ax
        scenario_entry.draw_to_ax(ax)

    def on_next_step(self):
        # TODO очищаем всю картинку и лог
        # TODO рисуем заново сигнал
        scenario_entries = self.scenario.get_entres_list_of_ith_step(self.i)

        for scenario_entry in scenario_entries:
            self.draw_entry_in_GUI(scenario_entry)
            # TODO добавить в лог scenario_entry.text_to_log