from settings import FREQUENCY, LEADS_NAMES, POINTS_TYPES, WAVES_TYPES,  MAX_SIGNAL_LEN
from neural_networks.neural_networks_helpers import get_appliable

from neural_networks.neural_networks_helpers.helpers_CNN.one_CNN_get_activations_on_signal import get_activations_of_CNN_on_signal
from neural_networks.neural_networks_helpers.helpers_CNN.one_CNN_activations_to_delineation import get_delineation_from_activation_by_mean, get_delineation_from_activation_by_extremum_signal

from neural_networks.neural_networks_helpers.helpers_CNN.group_CNN_get_activations_on_signals import get_activations_of_group_CNN


from neural_networks.neural_networks_models import CNN



class Deciser:
    def __init__(self, signals,  leads_names):
        self.signals = signals
        self.leads_names = leads_names

        self.scene = Scene() # Пустая сцена
        self.history = SceneHistory() # Пустая история
        
        # QRS
        self.cnn_i_qrs = get_appliable('BinaryDataset_10000_i_QRS_PEAK_250_0319_011511')
        self.cnn_ii_qrs = get_appliable('BinaryDataset_10000_ii_QRS_PEAK_250_0320_195342')
        self.cnn_iii_qrs = get_appliable('BinaryDataset_10000_iii_QRS_PEAK_250_0320_210148')

        # P
        self.cnn_i_p = get_appliable('BinaryDataset_10000_i_P_PEAK_250_0319_005838')
        self.cnn_ii_p = get_appliable('BinaryDataset_10000_ii_P_PEAK_250_0320_191832')
        self.cnn_iii_p = get_appliable('BinaryDataset_10000_iii_P_PEAK_250_0320_203732')

        # T
        self.cnn_i_t = get_appliable('BinaryDataset_10000_i_T_PEAK_250_0319_004209')
        self.cnn_ii_t = get_appliable('BinaryDataset_10000_ii_T_PEAK_250_0319_013127')
        self.cnn_iii_t = get_appliable('BinaryDataset_10000_iii_T_PEAK_250_0320_201605')
        
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
        
        
        self.get_delineation_and_weights_qrs_p_t(threshold = 0.5)
        
        self.radius_evidence = 10
        

    def get_delineation_and_weights_qrs_p_t(self, threshold):
        
        # QRS
        self.activations_i_qrs = get_activations_of_CNN_on_signal(self.cnn_i_qrs, self.signals[0])
        self.activations_ii_qrs = get_activations_of_CNN_on_signal(self.cnn_ii_qrs, self.signals[1])
        self.activations_iii_qrs = get_activations_of_CNN_on_signal(self.cnn_iii_qrs, self.signals[2])

        
        self.delineation_i_qrs, self.delin_weights_i_qrs = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_i_qrs, self.signals[0])
        self.delineation_ii_qrs, self.delin_weights_ii_qrs = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_ii_qrs, self.signals[1])
        self.delineation_iii_qrs, self.delin_weights_iii_qrs = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_iii_qrs, self.signals[2])

        # # P
        # self.activations_i_p = get_activations_of_CNN_on_signal(self.cnn_i_p, self.signals[0])
        # self.activations_ii_p = get_activations_of_CNN_on_signal(self.cnn_ii_p, self.signals[1])
        # self.activations_iii_p = get_activations_of_CNN_on_signal(self.cnn_iii_p, self.signals[2])
        
        # self.delineation_i_p, self.delin_weights_i_p = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_i_p, self.signals[0])
        # self.delineation_ii_p, self.delin_weights_ii_p = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_ii_p, self.signals[1])
        # self.delineation_iii_p, self.delin_weights_iii_p = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_iii_p, self.signals[2])
        
        # # T
        # self.activations_i_t = get_activations_of_CNN_on_signal(self.cnn_i_t, self.signals[0])
        # self.activations_ii_t = get_activations_of_CNN_on_signal(self.cnn_ii_t, self.signals[1])
        # self.activations_iii_t = get_activations_of_CNN_on_signal(self.cnn_iii_t, self.signals[2])
        
        # self.delineation_i_t, self.delin_weights_i_t = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_i_t, self.signals[0])
        # self.delineation_ii_t, self.delin_weights_ii_t = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_ii_t, self.signals[1])
        # self.delineation_iii_t, self.delin_weights_iii_t = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_iii_t, self.signals[2])

        
        
        
    def rank_by_weight(self, weights, delineation, coord_range):
        """
        Находит координату самой уверенной точки в заданном диапазоне.
        :param weights: Список уверенностей.
        :param delineation: Список координат точек.
        :param coord_range: Диапазон координат (min, max).
        :return: Координаты точек отсортированные по уверености в диапазоне или None, если таких точек нет.
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
        
        # Возвращаем координаты отсортированные по уверености
        return sorted_points
        
    
    def Finding_all_QRS_PEAK(self):
        
        # Нахождение всех R пиков и добавление на сцену
        
        # Создаем словарь candidates, где ключ - координата пика R в первом отведении, значение - список уверенностей
        candidates = {coord: [weight] for coord, weight in zip(self.delineation_i_qrs, self.delin_weights_i_qrs)}
        
        
        # Проходим по координатам пиков R в первом отведении
        for main_coord in candidates:
             # Ищем ближайшие координаты во втором отведении и прибавляем их уверенности
             for coord, weight in zip(self.delineation_ii_qrs, self.delin_weights_ii_qrs):
                 if abs(coord - main_coord) <= self.radius_evidence:
                     candidates[main_coord].append(weight)
         
             # Ищем ближайшие координаты в третьем отведении и прибавляем их уверенности
             for coord, weight in zip(self.delineation_iii_qrs, self.delin_weights_iii_qrs):
                 if abs(coord - main_coord) <= self.radius_evidence:
                     candidates[main_coord].append(weight)
         
        # Вычисляем среднюю уверенность для каждого кандидата
        average_confidence = {coord: sum(weights) / len(weights) for coord, weights in candidates.items()}
        
        result_delineation_qrs = []
        result_evidence_qrs    = []
        
        mean_evidence = sum(average_confidence.values()) / len(average_confidence)
        
        ids = []

        # Выводим результат
        for coord, avg_weight in average_confidence.items():
            print(f"Coordinate: {coord}, Evidence: {avg_weight:.2f}")
            
            if avg_weight >= 0.8:
                result_delineation_qrs.append(coord)
                result_evidence_qrs.append(avg_weight)
                
                R = DelineationPoint(t=coord/FREQUENCY,
                                          lead_name=LEADS_NAMES.i,
                                          point_type=POINTS_TYPES.QRS_PEAK,
                                          sertainty=0.5)
                id1 = self.scene.add_object(R)
                ids.append(id1)
                
                
        self.history.add_entry(visibles=ids)
                
            
        return  result_delineation_qrs, result_evidence_qrs
        
       

    def run(self):
        
        
        # Добавление на сцену активаций QRS на отведении I
        activ_qrs_i = Activations(net_activations=self.activations_i_qrs,
                            activations_t=self.time_s,
                            color=POINTS_TYPES_COLORS[POINTS_TYPES.QRS_PEAK],
                            lead_name=LEADS_NAMES.i)
        
        id1 = self.scene.add_object(activ_qrs_i)
        self.history.add_entry(visibles=[id1])
                
        result_delineation_qrs, result_evidence_qrs = self.Finding_all_QRS_PEAK()

        firstR_delineation = result_delineation_qrs[0]
        
        for ind in range(len(result_delineation_qrs) - 1):
                    
            # Отображение двух соседних пиков R
            nextR_delineation = result_delineation_qrs[ind + 1]
            
            
            if nextR_delineation < firstR_delineation:
                break
        
            # Отображение облаков активаций группы волны T между двумя пиками R
            activ_group_t = Activations(net_activations=self.activations_t[firstR_delineation: nextR_delineation],
                                activations_t=self.time_s[firstR_delineation: nextR_delineation],
                                color=POINTS_TYPES_COLORS[POINTS_TYPES.T_PEAK],
                                lead_name=LEADS_NAMES.i)   
            id4 = self.scene.add_object(activ_group_t)
            self.history.add_entry(visibles=[id4])


            
            # Пик T
            win_T = self.rank_by_weight(self.delin_weights_t, self.delineation_t, (firstR_delineation, nextR_delineation))
            
            if win_T == None: 
                self.history.add_entry(invisibles=[id4])
                firstR_delineation = nextR_delineation
                continue
            
            win_delin_point_T = DelineationPoint(t=win_T[0][1]/FREQUENCY,
                                      lead_name=LEADS_NAMES.i,
                                      point_type=POINTS_TYPES.T_PEAK,
                                      sertainty=0.5)
            id5 = self.scene.add_object(win_delin_point_T)
            self.history.add_entry(visibles=[id5])



            # Отображение групповых активаций волны P между поставленным пиком T и R
            activ_group_p = Activations(net_activations=self.activations_p[int(win_delin_point_T.t*FREQUENCY): nextR_delineation],
                                activations_t=self.time_s[int(win_delin_point_T.t*FREQUENCY): nextR_delineation],
                                color=POINTS_TYPES_COLORS[POINTS_TYPES.P_PEAK],
                                lead_name=LEADS_NAMES.i)
            id6 = self.scene.add_object(activ_group_p)
            self.history.add_entry(visibles=[id6], invisibles=[id4])
            
            

            # Пик P
            win_P = self.rank_by_weight(self.delin_weights_p, self.delineation_p, (win_T[0][1], nextR_delineation))
            
            if win_P == None: 
                self.history.add_entry(invisibles=[id6])
                firstR_delineation = nextR_delineation
                continue

            win_delin_point_P = DelineationPoint(t=win_P[0][1]/FREQUENCY,
                                      lead_name=LEADS_NAMES.i,
                                      point_type=POINTS_TYPES.P_PEAK,
                                      sertainty=0.5)
            id7 = self.scene.add_object(win_delin_point_P)
            self.history.add_entry(visibles=[id7])
            self.history.add_entry(invisibles=[id6])


            
            firstR_delineation = nextR_delineation
            
        
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
    patient_id  = test_ids[15]
    # 15

    
    signals_list_mV, leads_names_list_mV = get_signals_by_id_several_leads_mV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)
    signals_list_mkV, leads_names_list_mkV = get_signals_by_id_several_leads_mkV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)


    deciser = Deciser(signals=signals_list_mkV,  leads_names=leads_names_list_mkV)
    scene, scene_history = deciser.run()

    # scene, scene_history = create_test_scene_and_history() # их надо взять из отработавшего Deciser
    ui = UI_MainForm(leads_names=leads_names_list_mV, signals=signals_list_mV, scene=scene, scene_history=scene_history)
    
 
