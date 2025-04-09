import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from settings import FREQUENCY, LEADS_NAMES, POINTS_TYPES, WAVES_TYPES,  MAX_SIGNAL_LEN, POINTS_TYPES_COLORS

from neural_networks.neural_networks_helpers import (get_delineation_from_activation_by_mean,
                                                     get_delineation_from_activation_by_extremum_signal,
                                                     get_activations_of_group_CNN,
                                                     get_activations_of_CNN_on_signal)

from neural_networks import CNN, load_best_net, get_appliable
from decision_maker.logic import Scene, SceneHistory, Activations, DelineationPoint, DelineationInterval, SearchInterval



class Deciser:
    def __init__(self):
        self.signals = None
        self.leads_names = None

        self.scene = Scene() # Пустая сцена
        self.history = SceneHistory() # Пустая история
        
        # QRS
        self.cnn_i_qrs = load_best_net(POINTS_TYPES.QRS_PEAK, LEADS_NAMES.i)
        self.cnn_ii_qrs = load_best_net(POINTS_TYPES.QRS_PEAK, LEADS_NAMES.ii)
        self.cnn_iii_qrs = load_best_net(POINTS_TYPES.QRS_PEAK, LEADS_NAMES.iii)

        # P
        self.cnn_i_p = load_best_net(POINTS_TYPES.P_PEAK, LEADS_NAMES.i)
        # self.cnn_ii_p = get_appliable("BinaryDataset_10000_ii_P_PEAK_250_0320_191832")
        self.cnn_ii_p = load_best_net(POINTS_TYPES.P_PEAK, LEADS_NAMES.ii)
        self.cnn_iii_p = load_best_net(POINTS_TYPES.P_PEAK, LEADS_NAMES.iii)

        # T
        self.cnn_i_t = load_best_net(POINTS_TYPES.T_PEAK, LEADS_NAMES.i)
        self.cnn_ii_t = load_best_net(POINTS_TYPES.T_PEAK, LEADS_NAMES.ii)
        self.cnn_iii_t = load_best_net(POINTS_TYPES.T_PEAK, LEADS_NAMES.iii)
        
        self.time_s = [i/FREQUENCY for i in range(5000)]
        
        
        self.activations_i_qrs = None
        self.activations_ii_qrs = None
        self.activations_iii_qrs = None

        
        self.delineation_i_qrs, self.delin_weights_i_qrs = None, None
        self.delineation_ii_qrs, self.delin_weights_ii_qrs = None, None
        self.delineation_iii_qrs, self.delin_weights_iii_qrs = None, None

        # P
        self.activations_i_p = None
        self.activations_ii_p = None
        self.activations_iii_p = None
        
        self.delineation_i_p, self.delin_weights_i_p = None, None
        self.delineation_ii_p, self.delin_weights_ii_p = None, None
        self.delineation_iii_p, self.delin_weights_iii_p = None, None
        
        # T
        self.activations_i_t = None
        self.activations_ii_t = None
        self.activations_iii_t = None
        
        self.delineation_i_t, self.delin_weights_i_t = None, None
        self.delineation_ii_t, self.delin_weights_ii_t = None, None
        self.delineation_iii_t, self.delin_weights_iii_t = None, None        
        
        

        
        self.radius_evidence = 20
        self.threshold_evidence_qrs = 0.5
        self.threshold_evidence_p = 0.2
        self.threshold_evidence_t = 0.2

        
    def clear_scene(self):
        self.scene.scene_objects_dict.clear()
        self.history.visibles_groups.clear()
        self.history.invisibles_groups.clear()

        
        
    def what_points_we_want(self):
        return {LEADS_NAMES.i: [POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK]}
        
    def get_delineation_and_weights_qrs_p_t(self, threshold):
        
        # QRS
        self.activations_i_qrs = get_activations_of_CNN_on_signal(self.cnn_i_qrs, self.signals[0])
        self.activations_ii_qrs = get_activations_of_CNN_on_signal(self.cnn_ii_qrs, self.signals[1])
        self.activations_iii_qrs = get_activations_of_CNN_on_signal(self.cnn_iii_qrs, self.signals[2])

        
        self.delineation_i_qrs, self.delin_weights_i_qrs = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_i_qrs, self.signals[0], is_QRS_PEAK = True)
        self.delineation_ii_qrs, self.delin_weights_ii_qrs = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_ii_qrs, self.signals[1], is_QRS_PEAK = True)
        self.delineation_iii_qrs, self.delin_weights_iii_qrs = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_iii_qrs, self.signals[2], is_QRS_PEAK = True)

        # P
        self.activations_i_p = get_activations_of_CNN_on_signal(self.cnn_i_p, self.signals[0])
        self.activations_ii_p = get_activations_of_CNN_on_signal(self.cnn_ii_p, self.signals[1])
        self.activations_iii_p = get_activations_of_CNN_on_signal(self.cnn_iii_p, self.signals[2])
        
        self.delineation_i_p, self.delin_weights_i_p = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_i_p, self.signals[0])
        self.delineation_ii_p, self.delin_weights_ii_p = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_ii_p, self.signals[1])
        self.delineation_iii_p, self.delin_weights_iii_p = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_iii_p, self.signals[2])
        
        # T
        self.activations_i_t = get_activations_of_CNN_on_signal(self.cnn_i_t, self.signals[0])
        self.activations_ii_t = get_activations_of_CNN_on_signal(self.cnn_ii_t, self.signals[1])
        self.activations_iii_t = get_activations_of_CNN_on_signal(self.cnn_iii_t, self.signals[2])
        
        self.delineation_i_t, self.delin_weights_i_t = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_i_t, self.signals[0])
        self.delineation_ii_t, self.delin_weights_ii_t = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_ii_t, self.signals[1])
        self.delineation_iii_t, self.delin_weights_iii_t = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_iii_t, self.signals[2])

        
        
        
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
        
        
    def Calculate_evidence(self, wave_type, main_lead):
        """
        Параметры:
            wave_type: тип волны (qrs, p, t)
            main_lead: отведение в котором поставлена точка (LEADS_NAMES.i, LEADS_NAMES.ii, LEADS_NAMES.iii)
        
        Возвращает:
            Кортеж (result_delineation, result_evidence)
            result_delineation - координаты точек, имеющие evidence > threshold_evidence
        """
        # Порог
        threshold_evidence = getattr(self, f'threshold_evidence_{wave_type}')
        
        # Получаем данные для main_lead отведения
        main_delineation = getattr(self, f'delineation_{main_lead}_{wave_type}')
        main_weights = getattr(self, f'delin_weights_{main_lead}_{wave_type}')
        
        # Создаем словарь кандидатов {координата: [веса]}
        candidates = {coord: [weight] for coord, weight in zip(main_delineation, main_weights)}
        
        # Список отведений (кроме основного)
        other_leads = [lead for lead in [LEADS_NAMES.i, LEADS_NAMES.ii, LEADS_NAMES.iii] if lead != main_lead]
        
        # Ищем согласованные точки в других отведениях
        for lead in other_leads:
            lead_delineation = getattr(self, f'delineation_{lead}_{wave_type}')
            lead_weights = getattr(self, f'delin_weights_{lead}_{wave_type}')
            
            for main_coord in candidates:
                flag = False

                for coord, weight in zip(lead_delineation, lead_weights):
                    if abs(coord - main_coord) <= self.radius_evidence:
                        candidates[main_coord].append(weight)
                        flag = True
                        
                if not flag:
                    candidates[main_coord].append(0)
                        
        
        # Вычисляем среднюю уверенность для каждого кандидата
        average_confidence = {
            coord: sum(weights) / len(weights) 
            for coord, weights in candidates.items()
        }
        
        result_delineation = []
        result_evidence = []
        ids = []
        
       

        
        # Фильтруем по порогу threshold_evidence
        for coord, avg_weight in average_confidence.items():

            if avg_weight >= threshold_evidence:
                result_delineation.append(coord)
                result_evidence.append(avg_weight)
                    
        return result_delineation, result_evidence
    
    def get_candidate_points(self):
        
        # Добавление на сцену активаций QRS на отведении I
        activ_qrs_i = Activations(net_activations=self.activations_i_qrs,
                            activations_t=self.time_s,
                            color=POINTS_TYPES_COLORS[POINTS_TYPES.QRS_PEAK],
                            lead_name=LEADS_NAMES.i)
        
        id1 = self.scene.add_object(activ_qrs_i)
        self.history.add_entry(visibles=[id1])
                
        # Добавление на сцену всех R пиков
        # result_delineation_qrs, result_evidence_qrs = self.Finding_all_QRS_PEAK()
        result_delineation_qrs, result_evidence_qrs = self.Calculate_evidence(WAVES_TYPES.QRS, LEADS_NAMES.i)
        ids = []
        for coord in result_delineation_qrs:
            R = DelineationPoint(t=coord/FREQUENCY,
                                      lead_name=LEADS_NAMES.i,
                                      point_type=POINTS_TYPES.QRS_PEAK,
                                      sertainty=0.5)
            id_ = self.scene.add_object(R)
            ids.append(id_)
            
        self.history.add_entry(visibles=ids, invisibles=[id1])

        
        # Пик T
        result_delineation_t_i, result_evidence_t_i = self.Calculate_evidence(WAVES_TYPES.T, LEADS_NAMES.i)
        result_delineation_t_ii, result_evidence_t_ii = self.Calculate_evidence(WAVES_TYPES.T, LEADS_NAMES.ii)
        result_delineation_t_iii, result_evidence_t_iii = self.Calculate_evidence(WAVES_TYPES.T, LEADS_NAMES.iii)

        result_delineation_t = []
        result_delineation_t.extend(result_delineation_t_i)
        result_delineation_t.extend(result_delineation_t_ii)
        result_delineation_t.extend(result_delineation_t_iii)
        
        result_evidence_t = []
        result_evidence_t.extend(result_evidence_t_i)
        result_evidence_t.extend(result_evidence_t_ii)
        result_evidence_t.extend(result_evidence_t_iii)

        
        
        # Пик P
        result_delineation_p_i, result_evidence_p_i = self.Calculate_evidence(WAVES_TYPES.P, LEADS_NAMES.i)
        result_delineation_p_ii, result_evidence_p_ii = self.Calculate_evidence(WAVES_TYPES.P, LEADS_NAMES.ii)
        result_delineation_p_iii, result_evidence_p_iii = self.Calculate_evidence(WAVES_TYPES.P, LEADS_NAMES.iii)

        result_delineation_p = []
        result_delineation_p.extend(result_delineation_p_i)
        result_delineation_p.extend(result_delineation_p_ii)
        result_delineation_p.extend(result_delineation_p_iii)
        
        result_evidence_p = []
        result_evidence_p.extend(result_evidence_p_i)
        result_evidence_p.extend(result_evidence_p_ii)
        result_evidence_p.extend(result_evidence_p_iii)
        
        
        return result_delineation_qrs, result_evidence_qrs, result_delineation_p, result_evidence_p, result_delineation_t, result_evidence_t
    
    def run(self, signals,  leads_names):
        self.signals = signals
        self.leads_names = leads_names

        self.get_delineation_and_weights_qrs_p_t(threshold=0.2)
        
        result_delineation_qrs, result_evidence_qrs, result_delineation_p, result_evidence_p, result_delineation_t, result_evidence_t = self.get_candidate_points()

        if len(result_delineation_qrs) != 0:
            firstR_delineation = result_delineation_qrs[0]
    
            
            for ind in range(len(result_delineation_qrs) - 1):
                        
                # Отображение двух соседних пиков R
                nextR_delineation = result_delineation_qrs[ind + 1]
                
                
                if nextR_delineation < firstR_delineation:
                    break
            
                # Отображение облаков активаций группы волны T между двумя пиками R
                activ_group_t = Activations(net_activations=self.activations_i_t[firstR_delineation: nextR_delineation],
                                    activations_t=self.time_s[firstR_delineation: nextR_delineation],
                                    color=POINTS_TYPES_COLORS[POINTS_TYPES.T_PEAK],
                                    lead_name=LEADS_NAMES.i)   
                id4 = self.scene.add_object(activ_group_t)
                self.history.add_entry(visibles=[id4])
    
    
                
                # Пик T            
                win_T = self.rank_by_weight(result_evidence_t, result_delineation_t, (firstR_delineation, nextR_delineation))
                
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
                activ_group_p = Activations(net_activations=self.activations_i_p[int(win_delin_point_T.t*FREQUENCY): nextR_delineation],
                                    activations_t=self.time_s[int(win_delin_point_T.t*FREQUENCY): nextR_delineation],
                                    color=POINTS_TYPES_COLORS[POINTS_TYPES.P_PEAK],
                                    lead_name=LEADS_NAMES.i)
                id6 = self.scene.add_object(activ_group_p)
                self.history.add_entry(visibles=[id6], invisibles=[id4])
                
                
    
                # Пик P
                win_P = self.rank_by_weight(result_evidence_p, result_delineation_p, (win_T[0][1], nextR_delineation))
                
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
    
    print(patient_id)
    # 15
    # 4
    # 12

    
    signals_list_mV, leads_names_list_mV = get_signals_by_id_several_leads_mV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)
    signals_list_mkV, leads_names_list_mkV = get_signals_by_id_several_leads_mkV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)


    deciser = Deciser()
    scene, scene_history = deciser.run(signals=signals_list_mkV,  leads_names=leads_names_list_mkV)
    
    # scene, scene_history = create_test_scene_and_history() # их надо взять из отработавшего Deciser
    ui = UI_MainForm(leads_names=leads_names_list_mV, signals=signals_list_mV, scene=scene, scene_history=scene_history)
    
 


