from scene import Scene
from scene_history import SceneHistory
from decision_maker import UI_MainForm


class Deciser:
    def __init__(self, signals,  leads_names):
        self.signals = signals
        self.leads_names = leads_names

        self.scene = Scene() # Пустая сцена
        self.history = SceneHistory() # Пустая история

        # загружаем нейросети...
        # запоняем сцену и историю

    def run(self):
        pass

if __name__ == "__main__":
    from decision_maker.logic import Scene, create_test_scene_and_history

    # какие отведения хотим показать
    from settings import LEADS_NAMES
    from datasets.LUDB_utils import get_signals_by_id_several_leads_mV, get_LUDB_data, get_some_test_patient_id
    leads_names =[LEADS_NAMES.i, LEADS_NAMES.ii, LEADS_NAMES.iii]

    LUDB_data = get_LUDB_data()
    patient_id = get_some_test_patient_id()
    signals_list, leads_names_list = get_signals_by_id_several_leads_mV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)

    #deciser = Deciser(signals=signals_list,  leads_names=leads_names_list)
    #desiser.run()

    scene, scene_history = create_test_scene_and_history() # их надо взять из отработавшего Deciser
    ui = UI_MainForm(leads_names=leads_names_list, signals=signals_list, scene=scene, scene_history=scene_history)






