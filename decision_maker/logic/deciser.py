from scene import Scene
from scene_history import SceneHistory


class Deciser:
    def __init__(self, signals,  leads_names):
        self.signals = signals
        self.leads_names = leads_names

        self.scene = Scene() # Пустая сцена
        self.history = SceneHistory() # Пустая история

        # загружаем нейросети...
        # запоняем сцену и историю









