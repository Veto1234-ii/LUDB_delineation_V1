import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from settings import FREQUENCY, LEADS_NAMES, POINTS_TYPES, WAVES_TYPES,  MAX_SIGNAL_LEN, POINTS_TYPES_COLORS

from neural_networks.neural_networks_helpers import (get_delineation_from_activation_by_mean,
                                                     get_delineation_from_activation_by_extremum_signal,
                                                     get_activations_of_group_CNN,
                                                     get_activations_of_CNN_on_signal)

from neural_networks import CNN, load_best_net, get_appliable
from decision_maker.logic import Scene, SceneHistory, Activations, DelineationPoint, DelineationInterval, SearchInterval


class Deciser_leads:
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
        self.threshold_evidence_qrs = 0.7
        self.threshold_evidence_p = 0.2
        self.threshold_evidence_t = 0.2

        
    def clear_scene(self):
        self.scene.scene_objects_dict.clear()
        self.history.visibles_groups.clear()
        self.history.invisibles_groups.clear()

        
        
    def what_points_we_want(self):
        return {LEADS_NAMES.i: [POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK],
                LEADS_NAMES.ii: [POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK],
                LEADS_NAMES.iii: [POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK]
                }
    
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
        
        # Фильтруем по порогу threshold_evidence
        for coord, avg_weight in average_confidence.items():

            if avg_weight >= threshold_evidence:
                result_delineation.append(coord)
                result_evidence.append(avg_weight)
                    
        return result_delineation, result_evidence
    
    def generate_valid_point_groups(self, lead1, lead2, lead3):
        
        """
        Генерирует все допустимые группы точек, где:
        - Группа из 1 точки: точка только в одном отведении (два других пусты).
        - Группа из 2 точек: точки в двух отведениях (третье пусто).
        - Группа из 3 точек: точки во всех трёх отведениях.
        """
   
        groups = []
        
        # Группы из 3 точек (если во всех трёх отведениях есть точки)
        if lead1 and lead2 and lead3:
            for p1 in lead1:
                for p2 in lead2:
                    for p3 in lead3:
                        groups.append([p1, p2, p3])
        
        # Группы из 2 точек (если точки ровно в двух отведениях)
        if lead1 and lead2 and not lead3:
            for p1 in lead1:
                for p2 in lead2:
                    groups.append([p1, p2])
        if lead1 and lead3 and not lead2:
            for p1 in lead1:
                for p3 in lead3:
                    groups.append([p1, p3])
        if lead2 and lead3 and not lead1:
            for p2 in lead2:
                for p3 in lead3:
                    groups.append([p2, p3])
        
        # Группы из 1 точки (если точка только в одном отведении)
        if lead1 and not lead2 and not lead3:
            groups.extend([[p1] for p1 in lead1])
        if lead2 and not lead1 and not lead3:
            groups.extend([[p2] for p2 in lead2])
        if lead3 and not lead1 and not lead2:
            groups.extend([[p3] for p3 in lead3])
        
        
        
        return groups
    
    def select_best_group(self, groups):
        
        # Если нет ни одной группы (все отведения пусты)
        if not groups:
            return None
        
        # Вычисляем дисперсию для каждой группы
        groups_with_variance = []
        for group in groups:
            variance = np.var(group)  # Дисперсия координат группы
            groups_with_variance.append({
                'group': group,
                'variance': variance
            })
        
        # Находим группу с минимальной дисперсией
        min_variance_group = min(groups_with_variance, key=lambda x: x['variance'])
        
        return min_variance_group['group']
    
    def complete_missing_delineation_points(self, group):

        if len(group) == 3:
            return group  # Ничего не меняем
        
        if len(group) == 2:
            # Третья точка = среднее двух имеющихся
            avg = np.mean(group)
            return group + [avg]
        
        if len(group) == 1:
            # Две недостающие = единственной точке
            return group * 3
    
    def put_candidates_QRS_Peak(self):
        
        result_delineation_qrs_i, result_evidence_qrs_i = self.Calculate_evidence(WAVES_TYPES.QRS, LEADS_NAMES.i)
        result_delineation_qrs_ii, result_evidence_qrs_ii = self.Calculate_evidence(WAVES_TYPES.QRS, LEADS_NAMES.ii)
        result_delineation_qrs_iii, result_evidence_qrs_iii = self.Calculate_evidence(WAVES_TYPES.QRS, LEADS_NAMES.iii)
        
        qrs = {LEADS_NAMES.i:result_delineation_qrs_i , LEADS_NAMES.ii:result_delineation_qrs_ii, LEADS_NAMES.iii:result_delineation_qrs_iii}
        
        for lead, delineation in qrs.items():
            ids_i = []
            for coord in delineation:
                R = DelineationPoint(t=coord/FREQUENCY,
                                          lead_name=lead,
                                          point_type=POINTS_TYPES.QRS_PEAK,
                                          sertainty=0.5)
                id_ = self.scene.add_object(R)
                ids_i.append(id_)
                
            self.history.add_entry(visibles=ids_i)
        
        return result_delineation_qrs_i, result_delineation_qrs_ii, result_delineation_qrs_iii
        
    def run(self, signals,  leads_names):
        
        self.signals = signals
        self.leads_names = leads_names

        self.get_delineation_and_weights_qrs_p_t(threshold=0.2)
        
        # Расстановка точек QRS_PEAK
        result_delineation_qrs_i, result_delineation_qrs_ii, result_delineation_qrs_iii = self.put_candidates_QRS_Peak()
        
        
        # Отображение облаков активаций группы волны T
        # activ_group_t_i = Activations(net_activations=self.activations_i_t,
        #                     activations_t=self.time_s,
        #                     color=POINTS_TYPES_COLORS[POINTS_TYPES.T_PEAK],
        #                     lead_name=LEADS_NAMES.i)   
        # id4 = self.scene.add_object(activ_group_t_i)
        # self.history.add_entry(visibles=[id4])
        
        # activ_group_t_ii = Activations(net_activations=self.activations_ii_t,
        #                     activations_t=self.time_s,
        #                     color=POINTS_TYPES_COLORS[POINTS_TYPES.T_PEAK],
        #                     lead_name=LEADS_NAMES.ii)   
        # id5 = self.scene.add_object(activ_group_t_ii)
        # self.history.add_entry(visibles=[id5])
        
        # activ_group_t_iii = Activations(net_activations=self.activations_iii_t,
        #                     activations_t=self.time_s,
        #                     color=POINTS_TYPES_COLORS[POINTS_TYPES.T_PEAK],
        #                     lead_name=LEADS_NAMES.iii)   
        # id6 = self.scene.add_object(activ_group_t_iii)
        # self.history.add_entry(visibles=[id6])
        
        
        
        for i in range(len(result_delineation_qrs_i) - 1):
            
            # I отведение
            r_start_i = result_delineation_qrs_i[i]
            r_end_i = result_delineation_qrs_i[i + 1]
            
            # II отведение
            if i >= len(result_delineation_qrs_ii):
                r_start_ii = r_start_i
                r_end_ii = r_end_i
            else:
                r_start_ii = result_delineation_qrs_ii[i]
                r_end_ii = result_delineation_qrs_ii[i + 1] if (i + 1) < len(result_delineation_qrs_ii) else r_end_i
            
            # III отведение
            if i >= len(result_delineation_qrs_iii):
                r_start_iii = r_start_i
                r_end_iii = r_end_i
            else:
                r_start_iii = result_delineation_qrs_iii[i]
                r_end_iii = result_delineation_qrs_iii[i + 1] if (i + 1) < len(result_delineation_qrs_iii) else r_end_i

            
            # Расстановка точек P_PEAK между двумя соседними точками QRS_PEAK
            
            # Получение кандидатов в трех отведениях
            candidates_p_i = [coord for coord in self.delineation_i_p if coord > r_start_i and coord < r_end_i]
            candidates_p_ii = [coord for coord in self.delineation_ii_p if coord > r_start_ii and coord < r_end_ii]
            candidates_p_iii = [coord for coord in self.delineation_iii_p if coord > r_start_iii and coord < r_end_iii]
            
            # Составление всевозможных кобинаций точек 
            groups_p = self.generate_valid_point_groups(candidates_p_i, candidates_p_ii, candidates_p_iii)
            
            # Выбор лучшей комбинации
            best_group_p = self.select_best_group(groups_p)
            
            if best_group_p is None:
                full_group_p = [r_end_i, r_end_ii, r_end_iii]
                
            else:
                # Хотя бы одна точка стоит
                # Добавление недостающих точек         
                full_group_p = self.complete_missing_delineation_points(best_group_p)     
                ids = []
                for i, lead in enumerate([LEADS_NAMES.i, LEADS_NAMES.ii, LEADS_NAMES.iii]):
                        
                    delin_point = DelineationPoint(t=full_group_p[i]/FREQUENCY,
                                              lead_name=lead,
                                              point_type=POINTS_TYPES.P_PEAK,
                                              sertainty=0.5)
                    
                    id_ = self.scene.add_object(delin_point)
                    ids.append(id_)
                    
                self.history.add_entry(visibles=ids)

            # Расстановка точек T_PEAK между P_PEAK и следующей точкой QRS_PEAK
            
            # Получение кандидатов в трех отведениях
            candidates_t_i = [coord for coord in self.delineation_i_t if coord > r_start_i and coord < full_group_p[0]]
            candidates_t_ii = [coord for coord in self.delineation_ii_t if coord > r_start_ii and coord < full_group_p[1]]
            candidates_t_iii = [coord for coord in self.delineation_iii_t if coord > r_start_iii and coord < full_group_p[2]]
            
            # Составление всевозможных кобинаций точек 
            groups_t = self.generate_valid_point_groups(candidates_t_i, candidates_t_ii, candidates_t_iii)
            
            # Выбор лучшей комбинации
            best_group_t = self.select_best_group(groups_t)
            
            
            if best_group_t is not None:
                # Добавление недостающих точек         
                full_group_t = self.complete_missing_delineation_points(best_group_t)
                
                ids = []
                for i, lead in enumerate([LEADS_NAMES.i, LEADS_NAMES.ii, LEADS_NAMES.iii]):
                        
                    delin_point = DelineationPoint(t=full_group_t[i]/FREQUENCY,
                                              lead_name=lead,
                                              point_type=POINTS_TYPES.T_PEAK,
                                              sertainty=0.5)
                    
                    id_ = self.scene.add_object(delin_point)
                    ids.append(id_)
                    
                self.history.add_entry(visibles=ids)
                
  
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
    patient_id  = test_ids[55]
    
    print(patient_id)
    # 15
    # 4
    # 12

    
    signals_list_mV, leads_names_list_mV = get_signals_by_id_several_leads_mV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)
    signals_list_mkV, leads_names_list_mkV = get_signals_by_id_several_leads_mkV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)


    deciser = Deciser_leads()
    scene, scene_history = deciser.run(signals=signals_list_mkV,  leads_names=leads_names_list_mkV)
    
    # scene, scene_history = create_test_scene_and_history() # их надо взять из отработавшего Deciser
    ui = UI_MainForm(leads_names=leads_names_list_mV, signals=signals_list_mV, scene=scene, scene_history=scene_history)
    
 


        
        
        

