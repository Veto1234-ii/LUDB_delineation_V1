import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from settings import FREQUENCY, LEADS_NAMES, POINTS_TYPES, WAVES_TYPES,  MAX_SIGNAL_LEN, POINTS_TYPES_COLORS, TOLERANCE

from neural_networks.neural_networks_helpers import (get_delineation_from_activation_by_mean,
                                                     get_delineation_from_activation_by_extremum_signal,
                                                     get_activations_of_group_CNN,
                                                     get_activations_of_CNN_on_signal)

from neural_networks import CNN, load_best_net, get_appliable
from decision_maker.logic import Scene, SceneHistory, Activations, DelineationPoint, DelineationInterval, SearchInterval

import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict


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
        
        

        
        self.radius_evidence = 10
        self.threshold_evidence_qrs = 0.9
        self.threshold_evidence_p = 0.2
        self.threshold_evidence_t = 0.2


    def _find_neighbors(self, tree: KDTree, point: float):
        """Находит соседей точки в заданном радиусе"""
        neighbors = tree.query_ball_point(point, r=self.radius_evidence)
        if isinstance(neighbors, (int, np.integer)):
            return [neighbors]
        return list(neighbors)

    def find_evidence_groups(self, wave_type):
        # Подготовка данных
        leads = {
            LEADS_NAMES.i: (getattr(self, f'delineation_i_{wave_type}'), getattr(self, f'delin_weights_i_{wave_type}')),
            LEADS_NAMES.ii: (getattr(self, f'delineation_ii_{wave_type}'), getattr(self, f'delin_weights_ii_{wave_type}')),
            LEADS_NAMES.iii: (getattr(self, f'delineation_iii_{wave_type}'), getattr(self, f'delin_weights_iii_{wave_type}'))
        }
        
        # Построение KD-деревьев
        trees = {}
        points_data = {}
        for lead_name, (coords, weights) in leads.items():
            if coords.size == 0:
                continue
            trees[lead_name] = KDTree(coords.reshape(-1, 1))
            points_data[lead_name] = list(zip(coords, weights))
        
        evidence_groups = []
        used_points = set()  # Для отслеживания уже использованных точек

        # Поиск троек точек
        for lead1 in trees:
            other_leads = [lead for lead in trees if lead != lead1]
            if len(other_leads) < 2:
                continue
                
            lead2, lead3 = other_leads[0], other_leads[1]
            
            for idx1, (coord1, weight1) in enumerate(points_data[lead1]):
                # Пропускаем уже использованные точки
                if (lead1, idx1) in used_points:
                    continue
                    
                # Поиск во втором отведении
                neighbors2 = self._find_neighbors(trees[lead2], coord1)
                for idx2 in neighbors2:
                    if (lead2, idx2) in used_points:
                        continue
                        
                    coord2, weight2 = points_data[lead2][idx2]
                    
                    # Поиск в третьем отведении
                    neighbors3 = self._find_neighbors(trees[lead3], coord2)
                    for idx3 in neighbors3:
                        if (lead3, idx3) in used_points:
                            continue
                            
                        coord3, weight3 = points_data[lead3][idx3]
                        
                        # Проверка полного условия
                        if abs(coord1 - coord3) <= self.radius_evidence:
                            avg_weight = (weight1 + weight2 + weight3) / 3 
                            if avg_weight > getattr(self, f'threshold_evidence_{wave_type}'):
                                group = {
                                    lead1: (coord1, weight1),
                                    lead2: (coord2, weight2),
                                    lead3: (coord3, weight3),
                                    'average_weight': avg_weight
                                }
                                evidence_groups.append(group)
                                # Помечаем точки как использованные
                                used_points.add((lead1, idx1))
                                used_points.add((lead2, idx2))
                                used_points.add((lead3, idx3))
        
        # Сортировка по убыванию средней уверенности
        evidence_groups.sort(key=lambda x: -x['average_weight'])
        return evidence_groups
    
    def find_top_groups_in_range(self, groups, coord_range):
        min_coord, max_coord = coord_range
        
        filtered = [
            g for g in groups
            if any(min_coord <= g[lead][0] <= max_coord for lead in g if lead != 'average_weight')
        ]
        
        if not filtered:
            return None
            
        # Возвращаем группу с максимальным average_weight
        return max(filtered, key=lambda x: x['average_weight'])  

    def clear_scene(self):
        self.scene.scene_objects_dict.clear()
        self.history.visibles_groups.clear()
        self.history.invisibles_groups.clear()

        
        
    def what_points_we_want(self):
        return {LEADS_NAMES.i: [POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK],
                # LEADS_NAMES.ii: [POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK],
                # LEADS_NAMES.iii: [POINTS_TYPES.P_PEAK, POINTS_TYPES.QRS_PEAK, POINTS_TYPES.T_PEAK]
                }
        
    def get_delineation_and_weights_qrs_p_t(self, threshold):
        
        # QRS
        self.activations_i_qrs = get_activations_of_CNN_on_signal(self.cnn_i_qrs, self.signals[0])
        self.activations_ii_qrs = get_activations_of_CNN_on_signal(self.cnn_ii_qrs, self.signals[1])
        self.activations_iii_qrs = get_activations_of_CNN_on_signal(self.cnn_iii_qrs, self.signals[2])

        
        self.delineation_i_qrs, self.delin_weights_i_qrs = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_i_qrs, self.signals[0])
        self.delineation_ii_qrs, self.delin_weights_ii_qrs = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_ii_qrs, self.signals[1])
        self.delineation_iii_qrs, self.delin_weights_iii_qrs = get_delineation_from_activation_by_extremum_signal(threshold, self.activations_iii_qrs, self.signals[2])

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

        
        
        
  
    
    def run(self, signals,  leads_names):
        self.signals = signals
        self.leads_names = leads_names

        self.get_delineation_and_weights_qrs_p_t(threshold=0.2)
        
        
        
        groups_qrs = self.find_evidence_groups("qrs")
        groups_p = self.find_evidence_groups("p")
        groups_t = self.find_evidence_groups("t")
        
        print(groups_p)
        print()
        print(groups_t)
        
        ids_qrs = []
        ids_t = []
        ids_p = []
        
        # Сортируем QRS группы по координате первого отведения
        sorted_qrs_groups = sorted(groups_qrs, key=lambda g: g[list(g.keys())[0]][0])
        
        # Проходим по всем парам соседних QRS групп
        for i in range(len(sorted_qrs_groups)-1):
            current_qrs = sorted_qrs_groups[i]
            next_qrs = sorted_qrs_groups[i+1]
            
            # Получаем координаты текущего и следующего QRS
            current_coord = current_qrs[list(current_qrs.keys())[0]][0]
            next_coord = next_qrs[list(next_qrs.keys())[0]][0]
            
            # Создаем точки QRS для текущей группы
            for lead_name, value in current_qrs.items():
                if lead_name == 'average_weight':
                    continue
                    
                coord = value[0]
                point = DelineationPoint(
                    t=coord/FREQUENCY,
                    lead_name=lead_name,
                    point_type=POINTS_TYPES.QRS_PEAK,
                    sertainty=0.5  
                )
                ids_qrs.append(self.scene.add_object(point))
            
            # Ищем T-волну между текущим и следующим QRS
            t_range = (current_coord + 1, next_coord - 1)  # +1/-1 чтобы не пересекаться с QRS
            best_t_group = self.find_top_groups_in_range(groups_t, t_range)
            
            if best_t_group:
                # Создаем точки T для найденной группы
                for lead_name, value in best_t_group.items():
                    if lead_name == 'average_weight':
                        continue
                        
                    coord = value[0]
                    point = DelineationPoint(
                        t=coord/FREQUENCY,
                        lead_name=lead_name,
                        point_type=POINTS_TYPES.T_PEAK,
                        sertainty=0.5  
                    )
                    ids_t.append(self.scene.add_object(point))
                    
                
                # Ищем P-волну между T волной и следующим QRS
                p_range = (coord + 1, next_coord - 1)  # +1/-1 чтобы не пересекаться с QRS
                best_p_group = self.find_top_groups_in_range(groups_p, p_range)
                
                if best_p_group:
                    # Создаем точки T для найденной группы
                    for lead_name, value in best_p_group.items():
                        if lead_name == 'average_weight':
                            continue
                            
                        coord = value[0]
                        point = DelineationPoint(
                            t=coord/FREQUENCY,
                            lead_name=lead_name,
                            point_type=POINTS_TYPES.P_PEAK,
                            sertainty=0.5  
                        )
                        ids_p.append(self.scene.add_object(point))
                
        
        # Добавляем последнюю QRS группу (для которой нет следующей)
        last_qrs = sorted_qrs_groups[-1]
        for lead_name, value in last_qrs.items():
            if lead_name == 'average_weight':
                continue
                
            coord = value[0]
            point = DelineationPoint(
                t=coord/FREQUENCY,
                lead_name=lead_name,
                point_type=POINTS_TYPES.QRS_PEAK,
                sertainty=0.5
            )
            ids_qrs.append(self.scene.add_object(point))
        
        # Обновляем историю с видимыми точками
        self.history.add_entry(visibles=ids_qrs)
        self.history.add_entry(visibles=ids_p)
        self.history.add_entry(visibles=ids_t)

        
        
        
      


     
        
        # for ind in range(len(result_delineation_qrs) - 1):
                    
        #     # Отображение двух соседних пиков R
        #     nextR_delineation = result_delineation_qrs[ind + 1]
            
            
        #     if nextR_delineation < firstR_delineation:
        #         break
        
        #     # Отображение облаков активаций группы волны T между двумя пиками R
        #     activ_group_t = Activations(net_activations=self.activations_i_t[firstR_delineation: nextR_delineation],
        #                         activations_t=self.time_s[firstR_delineation: nextR_delineation],
        #                         color=POINTS_TYPES_COLORS[POINTS_TYPES.T_PEAK],
        #                         lead_name=LEADS_NAMES.i)   
        #     id4 = self.scene.add_object(activ_group_t)
        #     self.history.add_entry(visibles=[id4])


            
        #     # Пик T            
        #     win_T = self.rank_by_weight(result_evidence_t, result_delineation_t, (firstR_delineation, nextR_delineation))
            
        #     if win_T == None: 
        #         self.history.add_entry(invisibles=[id4])
        #         firstR_delineation = nextR_delineation
        #         continue
            
        #     win_delin_point_T = DelineationPoint(t=win_T[0][1]/FREQUENCY,
        #                               lead_name=LEADS_NAMES.i,
        #                               point_type=POINTS_TYPES.T_PEAK,
        #                               sertainty=0.5)
        #     id5 = self.scene.add_object(win_delin_point_T)
        #     self.history.add_entry(visibles=[id5])



        #     # Отображение групповых активаций волны P между поставленным пиком T и R
        #     activ_group_p = Activations(net_activations=self.activations_i_p[int(win_delin_point_T.t*FREQUENCY): nextR_delineation],
        #                         activations_t=self.time_s[int(win_delin_point_T.t*FREQUENCY): nextR_delineation],
        #                         color=POINTS_TYPES_COLORS[POINTS_TYPES.P_PEAK],
        #                         lead_name=LEADS_NAMES.i)
        #     id6 = self.scene.add_object(activ_group_p)
        #     self.history.add_entry(visibles=[id6], invisibles=[id4])
            
            

        #     # Пик P
        #     win_P = self.rank_by_weight(result_evidence_p, result_delineation_p, (win_T[0][1], nextR_delineation))
            
        #     if win_P == None: 
        #         self.history.add_entry(invisibles=[id6])
        #         firstR_delineation = nextR_delineation
        #         continue

        #     win_delin_point_P = DelineationPoint(t=win_P[0][1]/FREQUENCY,
        #                               lead_name=LEADS_NAMES.i,
        #                               point_type=POINTS_TYPES.P_PEAK,
        #                               sertainty=0.5)
        #     id7 = self.scene.add_object(win_delin_point_P)
        #     self.history.add_entry(visibles=[id7])
        #     self.history.add_entry(invisibles=[id6])


            
        #     firstR_delineation = nextR_delineation
            
            
        
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
    patient_id  = test_ids[17]
    
    print(patient_id)
    # 17

    
    signals_list_mV, leads_names_list_mV = get_signals_by_id_several_leads_mV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)
    signals_list_mkV, leads_names_list_mkV = get_signals_by_id_several_leads_mkV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)


    deciser = Deciser()
    scene, scene_history = deciser.run(signals=signals_list_mkV,  leads_names=leads_names_list_mkV)
    
    # our_coords = scene.get_binary_delineation_by_point_type_and_lead(LEADS_NAMES.i,
    #                                                                  POINTS_TYPES.T_PEAK)  # dвремя не в секундах

    # # вытаскиваем верную разметку этой точки в этом отведении
    # true_coords_in_seconds = get_one_lead_delineation_by_patient_id(patient_id=patient_id,
    #                                                      LUDB_data=LUDB_data,
    #                                                      lead_name=LEADS_NAMES.i,
    #                                                      point_type=POINTS_TYPES.T_PEAK)
    # true_coords = [true_coords_in_seconds[i]* FREQUENCY for i in range(len(true_coords_in_seconds))]
    
    # print(our_coords)
    # print(true_coords)
    
    # scene, scene_history = create_test_scene_and_history() # их надо взять из отработавшего Deciser
    ui = UI_MainForm(leads_names=leads_names_list_mV, signals=signals_list_mV, scene=scene, scene_history=scene_history)
    
 
