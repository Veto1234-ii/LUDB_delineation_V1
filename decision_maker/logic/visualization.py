import os

import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'


from settings import FREQUENCY, LEADS_NAMES, POINTS_TYPES, WAVES_TYPES,  MAX_SIGNAL_LEN, POINTS_TYPES_COLORS, TOLERANCE

from neural_networks.neural_networks_helpers import (get_delineation_from_activation_by_mean,
                                                     get_delineation_from_activation_by_extremum_signal,
                                                     get_activations_of_group_CNN,
                                                     get_activations_of_CNN_on_signal)

from neural_networks import CNN, load_best_net, get_appliable
from decision_maker.logic import Scene, SceneHistory, Activations, DelineationPoint, DelineationInterval, SearchInterval

class Visualization:
    def __init__(self):
        self.signals = None
        self.leads_names = None

        self.scene = Scene() # Пустая сцена
        self.history = SceneHistory() # Пустая история

        # T
        self.cnn_i_t = torch.load("D:\PycharmProjects\LUDB_delineation_V1\SAVED_NETS\BinaryDataset_5000_i_T_PEAK_500_0415_013605.pth", weights_only=False)
        self.cnn_ii_t = torch.load("D:\PycharmProjects\LUDB_delineation_V1\SAVED_NETS\BinaryDataset_5000_ii_T_PEAK_500_0415_010103.pth", weights_only=False)
        self.cnn_iii_t = torch.load("D:\PycharmProjects\LUDB_delineation_V1\SAVED_NETS\BinaryDataset_5000_iii_T_PEAK_500_0415_012547.pth", weights_only=False)
    
        self.activations_i_t = None
        self.activations_ii_t = None
        self.activations_iii_t = None
        
        self.time_s = [i/FREQUENCY for i in range(5000)]
        
    def run(self, signals,  leads_names):
        
        self.signals = signals
        self.leads_names = leads_names
        
        self.activations_i_t = get_activations_of_CNN_on_signal(self.cnn_i_t, self.signals[0])
        self.activations_ii_t = get_activations_of_CNN_on_signal(self.cnn_ii_t, self.signals[1])
        self.activations_iii_t = get_activations_of_CNN_on_signal(self.cnn_iii_t, self.signals[2])
        
        
        # Отображение облаков активаций волны T 
        activ_group_t_i = Activations(net_activations=self.activations_i_t,
                            activations_t=self.time_s,
                            color=POINTS_TYPES_COLORS[POINTS_TYPES.T_PEAK],
                            lead_name=LEADS_NAMES.i)   
        id4 = self.scene.add_object(activ_group_t_i)
    
    
        activ_group_t_ii = Activations(net_activations=self.activations_ii_t,
                            activations_t=self.time_s,
                            color=POINTS_TYPES_COLORS[POINTS_TYPES.T_PEAK],
                            lead_name=LEADS_NAMES.ii)   
        id5 = self.scene.add_object(activ_group_t_ii)
    
    
        activ_group_t_iii = Activations(net_activations=self.activations_iii_t,
                            activations_t=self.time_s,
                            color=POINTS_TYPES_COLORS[POINTS_TYPES.T_PEAK],
                            lead_name=LEADS_NAMES.iii)   
        id6 = self.scene.add_object(activ_group_t_iii)
        
        self.history.add_entry(visibles=[id4, id5, id6])

        return self.scene, self.history
    
if __name__ == "__main__":
    
    from decision_maker import UI_MainForm

    from decision_maker.logic import Scene, create_test_scene_and_history
    
    from decision_maker.logic.scene import Scene
    from decision_maker.logic.scene_history import SceneHistory

    from decision_maker.logic import DelineationPoint, DelineationInterval, SearchInterval, Activations

    # какие отведения хотим показать
    from settings import LEADS_NAMES, POINTS_TYPES_COLORS
    from datasets.LUDB_utils import get_signals_by_id_several_leads_mV, get_signals_by_id_several_leads_mkV, get_LUDB_data, get_some_test_patient_id, get_test_and_train_ids, get_signal_by_id_and_lead_mkV, get_one_lead_delineation_by_patient_id  
    leads_names =[LEADS_NAMES.i, LEADS_NAMES.ii, LEADS_NAMES.iii]

    LUDB_data = get_LUDB_data()
    
    train_ids, test_ids = get_test_and_train_ids(LUDB_data)
    patient_id  = test_ids[24]
    
    print(patient_id)
    # 17

    
    signals_list_mV, leads_names_list_mV = get_signals_by_id_several_leads_mV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)
    signals_list_mkV, leads_names_list_mkV = get_signals_by_id_several_leads_mkV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)


    viz = Visualization()
    scene, scene_history = viz.run(signals=signals_list_mkV,  leads_names=leads_names_list_mkV)
    
   
    
    ui = UI_MainForm(leads_names=leads_names_list_mV, signals=signals_list_mV, scene=scene, scene_history=scene_history)
    
        
        

