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
        
             
        self.time_s = [i/FREQUENCY for i in range(5000)]
        

        self.radius_evidence = 20
        self.threshold_evidence_QRS_PEAK = 0.7
        
        
        self.Coordinates_points_from_R_to_R = {}
    
        wave_types = [WAVES_TYPES.QRS, WAVES_TYPES.P, WAVES_TYPES.T]
        sub_types = ['_start', '_peak', '_end']
        
        for lead in [LEADS_NAMES.i, LEADS_NAMES.ii, LEADS_NAMES.iii]:
            self.Coordinates_points_from_R_to_R[lead] = {}
            
            for wave in wave_types:
                for sub in sub_types:
                    key = f"{wave}{sub}".upper()
                    self.Coordinates_points_from_R_to_R[lead][key] = None
     
        self.Types_points = {}
        
        for wave in wave_types:
            self.Types_points[wave] = [f"{wave}{sub}".upper() for sub in sub_types]
        
    
        for lead in self.Coordinates_points_from_R_to_R.keys():
                        
            for type_point in self.Coordinates_points_from_R_to_R[lead].keys():
                
                net = load_best_net(getattr(POINTS_TYPES, type_point), lead)
                
                setattr(self, f"cnn_{lead}_{type_point}", net)
            

        
    def clear_scene(self):
        self.scene.scene_objects_dict.clear()
        self.history.visibles_groups.clear()
        self.history.invisibles_groups.clear()

    def clear_coordinates(self):
        for lead, types_points in self.Coordinates_points_from_R_to_R.items():
            for type_point in types_points.keys():
                types_points[type_point] = None
        
    def what_points_we_want(self):
        return {LEADS_NAMES.i: [POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK],
                LEADS_NAMES.ii: [POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK],
                LEADS_NAMES.iii: [POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK]
                }
    
    def get_delineation_and_weights_qrs_p_t(self, threshold):
                
        
        for i, lead in enumerate(self.Coordinates_points_from_R_to_R.keys()):
                        
            for type_point in self.Coordinates_points_from_R_to_R[lead].keys():
                
                net = getattr(self, f"cnn_{lead}_{type_point}")
                
                activations = get_activations_of_CNN_on_signal(net, self.signals[i])
                
                if type_point == self.Types_points[WAVES_TYPES.QRS][1]:
                    is_QRS_PEAK = True
                else:
                    is_QRS_PEAK = False
                
                delineation, weight = get_delineation_from_activation_by_extremum_signal(threshold, activations, self.signals[i], is_QRS_PEAK)

                setattr(self, f"activations_{lead}_{type_point}", activations)
                setattr(self, f"delineation_{lead}_{type_point}", delineation)
                setattr(self, f"delin_weights_{lead}_{type_point}", weight)

        
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
        other_leads = [lead for lead in self.leads_names if lead != main_lead]
        
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
    
    def put_QRS_Peak(self):
        
        qrs_peak = self.Types_points[WAVES_TYPES.QRS][1]
        
        qrs = {}
        
        for lead in self.leads_names:
            qrs[lead], _ = self.Calculate_evidence(qrs_peak, lead)
        
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
        
        return qrs[self.leads_names[0]], qrs[self.leads_names[1]], qrs[self.leads_names[2]]
    
    def Put_best_group(self, start_idx, end_idx, type_point):
        
        # Получение кандидатов в трех отведениях
        
        delineation_i = getattr(self, f'delineation_i_{type_point}')
        delineation_ii = getattr(self, f'delineation_ii_{type_point}')
        delineation_iii = getattr(self, f'delineation_iii_{type_point}')

        
        candidates_i = [coord for coord in delineation_i if coord > start_idx[0] and coord < end_idx[0]]
        candidates_ii = [coord for coord in delineation_ii if coord > start_idx[1] and coord < end_idx[1]]
        candidates_iii = [coord for coord in delineation_iii if coord > start_idx[2] and coord < end_idx[2]]
        
        # Составление всевозможных кобинаций точек 
        groups = self.generate_valid_point_groups(candidates_i, candidates_ii, candidates_iii)
        
        # Выбор лучшей комбинации
        best_group = self.select_best_group(groups)
        
        if best_group is not None:
            
            # Хотя бы одна точка стоит
            # Добавление недостающих точек         
            full_group = self.complete_missing_delineation_points(best_group)     
            ids = []
            for i, lead in enumerate(self.leads_names):
                
                self.Coordinates_points_from_R_to_R[lead][type_point] = full_group[i]

                delin_point = DelineationPoint(t=full_group[i]/FREQUENCY,
                                          lead_name=lead,
                                          point_type=getattr(POINTS_TYPES, type_point),
                                          sertainty=0.5)
                
                id_ = self.scene.add_object(delin_point)
                ids.append(id_)
                
            self.history.add_entry(visibles=ids)
            
            
            
    def get_start_and_end_idx(self, r_start, r_end, type_point_start, type_point_end, type_point):
        
        if self.Coordinates_points_from_R_to_R[self.leads_names[0]][type_point_start] is None:
            
             start_idx = r_start
        else:
            
            start_i = self.Coordinates_points_from_R_to_R[self.leads_names[0]][type_point_start] 
            start_ii = self.Coordinates_points_from_R_to_R[self.leads_names[1]][type_point_start]  
            start_iii = self.Coordinates_points_from_R_to_R[self.leads_names[2]][type_point_start]

            start_idx = (start_i, start_ii, start_iii)
             
             
         
        if self.Coordinates_points_from_R_to_R[self.leads_names[0]][type_point_end]  is None:
             
             end_idx = r_end
        else:
             end_i = self.Coordinates_points_from_R_to_R[self.leads_names[0]][type_point_end] 
             end_ii = self.Coordinates_points_from_R_to_R[self.leads_names[1]][type_point_end]  
             end_iii = self.Coordinates_points_from_R_to_R[self.leads_names[2]][type_point_end]
             
             end_idx = (end_i, end_ii, end_iii)
        

        return start_idx, end_idx, type_point  
        
        
    def run(self, signals, leads_names):
        
        self.leads_names = leads_names
        self.signals = signals

        self.get_delineation_and_weights_qrs_p_t(threshold=0.5)
        
        # Расстановка точек QRS_PEAK
        result_delineation_qrs_i, result_delineation_qrs_ii, result_delineation_qrs_iii = self.put_QRS_Peak()      
        
        
        for i in range(len(result_delineation_qrs_i) - 1):
            
            self.clear_coordinates()
                        
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

            qrs_start = self.Types_points[WAVES_TYPES.QRS][0]
            qrs_end = self.Types_points[WAVES_TYPES.QRS][2]
            p_start = self.Types_points[WAVES_TYPES.P][0]
            p_end = self.Types_points[WAVES_TYPES.P][2]
            p_peak = self.Types_points[WAVES_TYPES.P][1]
            t_start = self.Types_points[WAVES_TYPES.T][0]
            t_peak = self.Types_points[WAVES_TYPES.T][1]
            t_end = self.Types_points[WAVES_TYPES.T][2]
    

            # QRS_START
            start_idx, end_idx, type_point  = self.get_start_and_end_idx(r_start = (r_start_i, r_start_ii, r_start_iii),
                                r_end = (r_end_i, r_end_ii, r_end_iii),
                                type_point_start = qrs_start,
                                type_point_end = qrs_start,
                                type_point = qrs_start)
            
            self.Put_best_group(start_idx, end_idx, type_point)
            
            # QRS_END
            start_idx, end_idx, type_point  = self.get_start_and_end_idx(r_start = (r_start_i, r_start_ii, r_start_iii),
                                r_end = (r_end_i, r_end_ii, r_end_iii),
                                type_point_start = qrs_end,
                                type_point_end = qrs_start,
                                type_point = qrs_end)
            self.Put_best_group(start_idx, end_idx, type_point)

            
            # P_PEAK
            start_idx, end_idx, type_point  = self.get_start_and_end_idx(r_start = (r_start_i, r_start_ii, r_start_iii),
                                r_end = (r_end_i, r_end_ii, r_end_iii),
                                type_point_start = qrs_end,
                                type_point_end = qrs_start,
                                type_point = p_peak)
            self.Put_best_group(start_idx, end_idx, type_point)

            # P_START
            start_idx, end_idx, type_point  = self.get_start_and_end_idx(r_start = (r_start_i, r_start_ii, r_start_iii),
                                r_end = (r_end_i, r_end_ii, r_end_iii),
                                type_point_start = qrs_end,
                                type_point_end = p_peak,
                                type_point = p_start)
            self.Put_best_group(start_idx, end_idx, type_point)

            # P_END
            start_idx, end_idx, type_point  = self.get_start_and_end_idx(r_start = (r_start_i, r_start_ii, r_start_iii),
                                r_end = (r_end_i, r_end_ii, r_end_iii),
                                type_point_start = p_peak,
                                type_point_end = qrs_start,
                                type_point = p_end)
            self.Put_best_group(start_idx, end_idx, type_point)

            # T_PEAK
            start_idx, end_idx, type_point  = self.get_start_and_end_idx(r_start = (r_start_i, r_start_ii, r_start_iii),
                                r_end = (r_end_i, r_end_ii, r_end_iii),
                                type_point_start = qrs_end,
                                type_point_end = p_start,
                                type_point = t_peak)
            self.Put_best_group(start_idx, end_idx, type_point)

            # T_START
            start_idx, end_idx, type_point  = self.get_start_and_end_idx(r_start = (r_start_i, r_start_ii, r_start_iii),
                                r_end = (r_end_i, r_end_ii, r_end_iii),
                                type_point_start = qrs_end,
                                type_point_end = t_peak,
                                type_point = t_start)
            self.Put_best_group(start_idx, end_idx, type_point)

            # T_END
            start_idx, end_idx, type_point  = self.get_start_and_end_idx(r_start = (r_start_i, r_start_ii, r_start_iii),
                                r_end = (r_end_i, r_end_ii, r_end_iii),
                                type_point_start = t_peak,
                                type_point_end = p_start,
                                type_point = t_end)
            
            self.Put_best_group(start_idx, end_idx, type_point)

            

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
    patient_id  = test_ids[12]
    
    print(patient_id)
    # 55
    # 4
    # 12

    
    signals_list_mV, leads_names_list_mV = get_signals_by_id_several_leads_mV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)
    signals_list_mkV, leads_names_list_mkV = get_signals_by_id_several_leads_mkV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)


    deciser = Deciser_leads()
    scene, scene_history = deciser.run(signals=signals_list_mkV, leads_names=leads_names_list_mkV)
    
    # scene, scene_history = create_test_scene_and_history() # их надо взять из отработавшего Deciser
    ui = UI_MainForm(leads_names=leads_names_list_mV, signals=signals_list_mV, scene=scene, scene_history=scene_history)
    
 


        
        
        



