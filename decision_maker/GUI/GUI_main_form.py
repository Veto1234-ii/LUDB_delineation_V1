from decision_maker.logic import Scene


class GUI_MainForm:
    def __init__(self, signals, leads_names, scene):
        self.scene = scene
        self.current_step_num = -1
        self.ax_list =
        # TODO создаем GUI - форму с матлотлиб fig для отрисовки отведений и текстовое окошко для лога
        # Рисуем в картинку сигналы
        # привязываем self.on_next_step в кач-ве обработчика к событию "клик по стрелочке вправо на клаивтуре"
        # а также и self.on_prev_step
        pass



    def on_next_step(self):
        # TODO очищаем всю картинку и лог
        # TODO рисуем заново сигнал
        self.current_step_num += 1
        self.scene.draw(ax_list=self.ax_list, num_steps=self.current_step_num)



