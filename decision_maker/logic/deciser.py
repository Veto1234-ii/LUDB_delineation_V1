from settings import FREQUENCY, LEADS_NAMES, POINTS_TYPES, WAVES_TYPES
from neural_networks_helpers import get_appliable

from neural_networks_helpers.helpers_CNN.one_CNN_get_activations_on_signal import get_activations_of_CNN_on_signal
from neural_networks_helpers.helpers_CNN.one_CNN_activations_to_delineation import get_delineation_from_activation_by_mean

from neural_networks_helpers.helpers_CNN.group_CNN_get_activations_on_signals import get_activations_of_group_CNN

from neural_networks_helpers.helpers_CNN.F1_of_CNN import get_F1_of_one_CNN


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
        
        self.time_s = [i/FREQUENCY for i in range(5000)]
        
        
        self.activations_qrs = None
        self.activations_p = None
        self.activations_t = None
        
        self.delineation_qrs = None
        self.delineation_p = None
        self.delineation_t = None
        
        self.delin_weights_qrs = None
        self.delin_weights_p = None
        self.delin_weights_t = None        


    def get_delineation_and_weights_qrs_p_t(self, threshold):
        
        self.activations_qrs = get_activations_of_CNN_on_signal(self.cnn_i_qrs, self.signals[0])

        self.delineation_qrs, self.delin_weights_qrs = get_delineation_from_activation_by_mean(threshold, self.activations_qrs)
        
        self.activations_p = get_activations_of_group_CNN([self.cnn_i_p, self.cnn_ii_p, self.cnn_iii_p],
                                                           [self.signals[0], self.signals[1], self.signals[2]])
        
        self.activations_t = get_activations_of_group_CNN([self.cnn_i_t, self.cnn_ii_t, self.cnn_iii_t],
                                                           [self.signals[0], self.signals[1], self.signals[2]])
        
        self.delineation_p, self.delin_weights_p = get_delineation_from_activation_by_mean(threshold, self.activations_p)
        self.delineation_t, self.delin_weights_t = get_delineation_from_activation_by_mean(threshold, self.activations_t)
        
    def find_most_confident_point(self, weights, delineation, coord_range):
        """
        Находит координату самой уверенной точки в заданном диапазоне.
    
        :param weights: Список уверенностей.
        :param delineation: Список координат точек.
        :param coord_range: Диапазон координат (min, max).
        :return: Координата самой уверенной точки в диапазоне или None, если таких точек нет.
        """
        # Объединяем weights и delineation в список кортежей
        combined = list(zip(weights, delineation))
        
        # Фильтруем точки, которые попадают в заданный диапазон
        filtered_points = [(w, coord) for w, coord in combined if coord_range[0] <= coord <= coord_range[1]]
        
        # Если нет точек в диапазоне, возвращаем None
        if not filtered_points:
            return None
        
        # Сортируем по уверенности (по убыванию)
        sorted_points = sorted(filtered_points, key=lambda x: x[0], reverse=True)
        
        # Возвращаем координату самой уверенной точки
        return sorted_points[0][1]
        
        
    def add_DelineationPoint_between(self, first, second, point_type, delineation):
        
        # Добавление на сцену DelineationPoint между двумя "отсчетами" сигнала
        for d in delineation:
            if (d > first) and (d < second):
                Point = DelineationPoint(t=d/FREQUENCY,
                                          lead_name=LEADS_NAMES.i,
                                          point_type=point_type,
                                          sertainty=0.5)
                
                id_Point = self.scene.add_object(Point)
                
    
     
    def run(self):
        
        self.get_delineation_and_weights_qrs_p_t(threshold = 0.8)
           
        # Добавление на сцену активаций QRS на отведении I
        activ_qrs = Activations(net_activations=self.activations_qrs,
                            activations_t=self.time_s,
                            color=POINTS_TYPES_COLORS[POINTS_TYPES.QRS_PEAK],
                            lead_name=LEADS_NAMES.i)
        
        id1 = self.scene.add_object(activ_qrs)
        self.history.add_entry(visibles=[id1])
        
       
        
        # Первый пик R
        firstR_delineation = self.delineation_qrs[0]
        firstR = DelineationPoint(t=firstR_delineation/FREQUENCY,
                                  lead_name=LEADS_NAMES.i,
                                  point_type=POINTS_TYPES.QRS_PEAK,
                                  sertainty=0.5)
        id2 = self.scene.add_object(firstR)
        self.history.add_entry(visibles=[id2], invisibles=[id1])
        

        # Цикл
        
        ind = 0
        while ind < len(self.delineation_qrs) - 1:
            
            # Отображение двух соседних пиков R
            nextR_delineation = self.delineation_qrs[ind + 1]

            nextR = DelineationPoint(t=nextR_delineation/FREQUENCY,
                                      lead_name=LEADS_NAMES.i,
                                      point_type=POINTS_TYPES.QRS_PEAK,
                                      sertainty=0.5)
            id3 = self.scene.add_object(nextR)
            self.history.add_entry(visibles=[id3])
            
        
            # Отображение облаков активаций группы волны T между двумя пиками R
            activ_group_t = Activations(net_activations=self.activations_t[firstR_delineation: nextR_delineation],
                                activations_t=self.time_s[firstR_delineation: nextR_delineation],
                                color=POINTS_TYPES_COLORS[POINTS_TYPES.T_PEAK],
                                lead_name=LEADS_NAMES.i)   
            id4 = self.scene.add_object(activ_group_t)
            self.history.add_entry(visibles=[id4])


            # Добавление на сцену DelineationPoint T_PEAK между двумя проставленными пиками R
            self.add_DelineationPoint_between_object_scene(firstR_delineation, nextR_delineation,
                                                           POINTS_TYPES.T_PEAK, self.delineation_t)
            
            find_most_confident_point(self, weights, delineation, coord_range)
                    
            # Поиск ближайшего справа пика T к пику R (firstR)
            nearest_delin_point_T = self.scene.get_nearest_delin_point(firstR_delineation/FREQUENCY,
                                                                     point_type = POINTS_TYPES.T_PEAK,
                                                                     lead_name = LEADS_NAMES.i,
                                                                     to_left=False)

            id5 = self.scene.add_object(nearest_delin_point_T)
            self.history.add_entry(visibles=[id5])
        
       

            # Отображение групповых активаций волны P между поставленным пиком T и R
            activ_group_p = Activations(net_activations=self.activations_p[int(nearest_delin_point_T.t*FREQUENCY): nextR_delineation],
                                activations_t=self.time_s[int(nearest_delin_point_T.t*FREQUENCY): nextR_delineation],
                                color=POINTS_TYPES_COLORS[POINTS_TYPES.P_PEAK],
                                lead_name=LEADS_NAMES.i)
            id6 = self.scene.add_object(activ_group_p)
            self.history.add_entry(visibles=[id6], invisibles=[id4])

        
            # Добавление на сцену DelineationPoint P_PEAK между проставленными T и R пиками   
            self.add_DelineationPoint_between_object_scene(int(nearest_delin_point_T.t*FREQUENCY), nextR_delineation,
                                                           POINTS_TYPES.P_PEAK, self.delineation_p)

                    
    
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
    from datasets.LUDB_utils import get_signals_by_id_several_leads_mV, get_signals_by_id_several_leads_mkV, get_LUDB_data, get_some_test_patient_id, get_test_and_train_ids, get_signal_by_id_and_lead_mkV, get_one_lead_delineation_by_patient_id  
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
    
    
    # model = get_appliable('BinaryDataset_1000_i_QRS_PEAK_250')
    
    # test_signals = []
    # true_delinations = []
    # for id_ in test_ids:
    #     test_signals.append(get_signal_by_id_and_lead_mkV(id_, 'i', LUDB_data))
    #     true_delinations.append([int(FREQUENCY*i) for i in get_one_lead_delineation_by_patient_id(id_, LUDB_data, LEADS_NAMES.i, POINTS_TYPES.QRS_PEAK)])
        
    # F1, mean_err = get_F1_of_one_CNN(model, test_signals, true_delinations, threshold=0.8, tolerance=25)
   
    # print(F1)
    # print(mean_err)






