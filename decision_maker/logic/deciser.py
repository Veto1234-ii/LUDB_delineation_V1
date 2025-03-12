from settings import FREQUENCY, LEADS_NAMES, POINTS_TYPES, WAVES_TYPES
from neural_networks_helpers import get_appliable

from neural_networks_helpers.helpers_CNN.one_CNN_get_activations_on_signal import get_activations_of_CNN_on_signal
from neural_networks_helpers.helpers_CNN.one_CNN_activations_to_delineation import get_delineation_from_activation_by_mean

from neural_networks_helpers.helpers_CNN.group_CNN_get_activations_on_signals import get_activations_of_group_CNN
from neural_networks_helpers.helpers_CNN.group_CNN_to_delineation import get_democracy_delineation_by_mean

from decision_maker.logic.scene import Scene
from decision_maker.logic.scene_history import SceneHistory
from decision_maker import UI_MainForm

from decision_maker.logic import DelineationPoint, DelineationInterval, SearchInterval, Activations

from neural_networks_models import CNN



class Deciser:
    def __init__(self, signals,  leads_names):
        self.signals = signals
        self.leads_names = leads_names

        self.scene = Scene() # Пустая сцена
        self.history = SceneHistory() # Пустая история
        
        # QRS
        self.cnn_i_qrs = get_appliable('cnn_i_qrs')
        self.cnn_ii_qrs = get_appliable('cnn_ii_qrs')
        self.cnn_iii_qrs = get_appliable('cnn_iii_qrs')

        # P
        self.cnn_i_p = get_appliable('cnn_i_p')
        self.cnn_ii_p = get_appliable('cnn_ii_p')
        self.cnn_iii_p = get_appliable('cnn_iii_p')

        # T
        self.cnn_i_t = get_appliable('cnn_i_t')
        self.cnn_ii_t = get_appliable('cnn_ii_t')
        self.cnn_iii_t = get_appliable('cnn_iii_t')


        
        
    def run(self):
        
        # Шаг 1 - Получение активаций QRS на отведении I
        activation_t = [i/FREQUENCY for i in range(5000)]
        
        activations_qrs_i = get_activations_of_CNN_on_signal(self.cnn_i_qrs, self.signals[0])
                
        activ_qrs = Activations(net_activations=activations_qrs_i,
                            activations_t=activation_t,
                            color=POINTS_TYPES_COLORS[POINTS_TYPES.QRS_PEAK],
                            lead_name=LEADS_NAMES.i)
        
        # Добавление на сцену активаций QRS на отведении I
        id1 = self.scene.add_object(activ_qrs)
        self.history.add_entry(visibles=[id1])
        
        # Шаг 2 - Получение разметки QRS на отведении I
        threshold = 0.8
        delineation_qrs_i = get_delineation_from_activation_by_mean(threshold, activations_qrs_i)
             
        
        # Шаг 3 - Получение активаций группы (отведения I, II, III) волн P и T
                
        activations_p_group = get_activations_of_group_CNN([self.cnn_i_p, self.cnn_ii_p, self.cnn_iii_p],
                                                           [self.signals[0], self.signals[1], self.signals[2]])
        
        activations_t_group = get_activations_of_group_CNN([self.cnn_i_t, self.cnn_ii_t, self.cnn_iii_t],
                                                           [self.signals[0], self.signals[1], self.signals[2]])
        
        # Шаг 4 - Получение разметки волн P и T       
        threshold = 0.8
        delineation_p_group = get_democracy_delineation_by_mean(threshold, activations_p_group)
        delineation_t_group = get_democracy_delineation_by_mean(threshold, activations_t_group)


        # Первый пик R
        firstR_delineation = delineation_qrs_i[0]
        
        firstR = DelineationPoint(t=firstR_delineation/FREQUENCY,
                                  lead_name=LEADS_NAMES.i,
                                  point_type=POINTS_TYPES.QRS_PEAK,
                                  sertainty=0.5)
        id2 = self.scene.add_object(firstR)


        self.history.add_entry(visibles=[id2], invisibles=[id1])

        # Цикл
        
        ind = 0
        while ind < len(delineation_qrs_i) - 1:
            
            # Отображение двух соседних пиков R
            nextR_delineation = delineation_qrs_i[ind + 1]

            
            
            nextR = DelineationPoint(t=nextR_delineation/FREQUENCY,
                                      lead_name=LEADS_NAMES.i,
                                      point_type=POINTS_TYPES.QRS_PEAK,
                                      sertainty=0.5)
            
            id3 = self.scene.add_object(nextR)
            
            self.history.add_entry(visibles=[id3])
        
            # Отображение облаков активаций группы волны T между двумя пиками R
        
            activ_group_t = Activations(net_activations=activations_t_group[firstR_delineation: nextR_delineation],
                                activations_t=activation_t[firstR_delineation: nextR_delineation],
                                color=POINTS_TYPES_COLORS[POINTS_TYPES.T_PEAK],
                                lead_name=LEADS_NAMES.i)   
            
            id4 = self.scene.add_object(activ_group_t)
            
            self.history.add_entry(visibles=[id4])


            # Добавление на сцену DelineationPoint T_PEAK между двумя пиками R
            for t_delin in delineation_t_group:
                
                if (t_delin > firstR_delineation) and (t_delin < nextR_delineation):
                    T_PEAK_point = DelineationPoint(t=t_delin/FREQUENCY,
                                              lead_name=LEADS_NAMES.i,
                                              point_type=POINTS_TYPES.T_PEAK,
                                              sertainty=0.5)
                    
                    id_T = self.scene.add_object(T_PEAK_point)
                    
            # Поиск ближайшего справа пика T к пику R (firstR)
            nearest_delin_point_T = self.scene.get_nearest_delin_point(firstR_delineation/FREQUENCY,
                                                                     point_type = POINTS_TYPES.T_PEAK,
                                                                     lead_name = LEADS_NAMES.i,
                                                                     to_left=False)

            id5 = self.scene.add_object(nearest_delin_point_T)
            self.history.add_entry(visibles=[id5])
        
       

            # Отображение групповых активаций волны P между поставленным пиком T и R
            activ_group_p = Activations(net_activations=activations_p_group[int(nearest_delin_point_T.t*FREQUENCY): nextR_delineation],
                                activations_t=activation_t[int(nearest_delin_point_T.t*FREQUENCY): nextR_delineation],
                                color=POINTS_TYPES_COLORS[POINTS_TYPES.P_PEAK],
                                lead_name=LEADS_NAMES.i)
            
            id6 = self.scene.add_object(activ_group_p)
            
            self.history.add_entry(visibles=[id6], invisibles=[id4])

        
            # Добавление на сцену DelineationPoint P_PEAK между поставленным T и R пиком       
            
            for p_delin in delineation_p_group:
                if p_delin > int(nearest_delin_point_T.t*FREQUENCY) and p_delin < nextR_delineation:
                    
                    P_PEAK_point = DelineationPoint(t=p_delin/FREQUENCY,
                                              lead_name=LEADS_NAMES.i,
                                              point_type=POINTS_TYPES.P_PEAK,
                                              sertainty=0.5)
                    
                    id_P = self.scene.add_object(P_PEAK_point)
                    
    
            # Поиск ближайшего слева пика P к пику R (nextR)
            nearest_delin_point_P = self.scene.get_nearest_delin_point(nextR_delineation/FREQUENCY,
                                                                     point_type = POINTS_TYPES.P_PEAK,
                                                                     lead_name = LEADS_NAMES.i,
                                                                     to_left=True)
            
            id7 = self.scene.add_object(nearest_delin_point_P)
            self.history.add_entry(visibles=[id7])
            
            
            
            self.history.add_entry(invisibles=[id6])
            
            firstR_delineation = nextR_delineation
            
            ind+=1
        
        return self.scene, self.history

if __name__ == "__main__":
    from decision_maker.logic import Scene, create_test_scene_and_history

    # какие отведения хотим показать
    from settings import LEADS_NAMES, POINTS_TYPES_COLORS
    from datasets.LUDB_utils import get_signals_by_id_several_leads_mV, get_signals_by_id_several_leads_mkV, get_LUDB_data, get_some_test_patient_id, get_test_and_train_ids
    leads_names =[LEADS_NAMES.i, LEADS_NAMES.ii, LEADS_NAMES.iii]

    LUDB_data = get_LUDB_data()
    # patient_id = get_some_test_patient_id()
    
    train_ids, test_ids = get_test_and_train_ids(LUDB_data)
    patient_id  = test_ids[10]
    # # 6
    # 14
    
    signals_list_mV, leads_names_list_mV = get_signals_by_id_several_leads_mV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)
    signals_list_mkV, leads_names_list_mkV = get_signals_by_id_several_leads_mkV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)


    deciser = Deciser(signals=signals_list_mkV,  leads_names=leads_names_list_mkV)
    scene, scene_history = deciser.run()

    # scene, scene_history = create_test_scene_and_history() # их надо взять из отработавшего Deciser
    ui = UI_MainForm(leads_names=leads_names_list_mV, signals=signals_list_mV, scene=scene, scene_history=scene_history)






