from .group_CNN_get_activations_on_signals import get_activations_of_group_CNN
from .group_CNN_to_delineation import get_democracy_delineation_by_mean
import numpy as np

def get_F1_of_group_CNN(trained_CNNs, signals, true_delinations, threshold, tolerance):
    """ signals - сигналы 12-ти отведений разных пациентов"""
    """ true_delinations - DocDelineation I отведения"""
    pairs = []
    
    TP = 0
    FP = 0
    FN = 0
    
    dist_border = 700
    
    for i in range(len(signals)):
        
        leads = signals[i]
        result_activations = get_activations_of_group_CNN(trained_CNNs, leads)
        delineation = get_democracy_delineation_by_mean(threshold, result_activations)
        
        true_delination = true_delinations[i]
        
                
        program_labels = np.array([i for i in delineation if (i >= dist_border) and (i <= len(leads[0]) - dist_border)])
        doctor_labels = np.array([i for i in true_delination if (i >= dist_border) and (i <= len(leads[0]) - dist_border)])

        if len(doctor_labels) == 0:
            continue

        TP_ = 0  # True Positives
        FP_ = 0  # False Positives
        FN_ = 0  # False Negatives

        # Копируем программную разметку, чтобы удалять использованные точки
        available_program_labels = program_labels.tolist()

        # Проверяем каждую точку докторской разметки
        for doctor_point in doctor_labels:
            # Находим ближайшую точку программной разметки в пределах толерантного окна
            min_distance = float('inf')
            closest_point = None

            for program_point in available_program_labels:
                distance = abs(program_point - doctor_point)
                if distance <= tolerance and distance < min_distance:
                    min_distance = distance
                    closest_point = program_point

            # Если найдена ближайшая точка, фиксируем её как TP и удаляем из доступных
            if closest_point is not None:
                TP_ += 1
                pairs.append((doctor_point, closest_point))
                available_program_labels.remove(closest_point)
            else:
                FN_ += 1

        # Оставшиеся точки программной разметки считаем как FP
        FP_ = len(available_program_labels)
                    
        TP += TP_
        FP += FP_
        FN += FN_
    
    # F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # mean_err
    total_distance = 0
    for doctor_point, program_point in pairs:
        distance = abs(doctor_point - program_point)
        total_distance+=distance
        
    mean_err = total_distance/len(pairs)
    
    return F1, mean_err
