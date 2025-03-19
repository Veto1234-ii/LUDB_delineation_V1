import numpy as np

def find_activation_clouds(threshold, activations):
    """
    Находит облака активаций на основе порога.
    Возвращает список облаков, где каждое облако — это кортеж (индексы, значения активаций).
    """
    activation_clouds = []

    current_cloud_x = []
    current_cloud_y = []
        
    
    for i in range(len(activations)-1):
        
        if activations[i] > threshold:
            
            current_cloud_y.append(activations[i])
            current_cloud_x.append(i)
            if activations[i+1] < threshold:
                
                activation_clouds.append((current_cloud_x.copy(), current_cloud_y.copy()))
                current_cloud_x.clear()
                current_cloud_y.clear()
                
    return activation_clouds


def find_extremum_coord(signal, activation_cloud_x):
    # 1. Извлечение сигнала, соответствующего облаку активаций
    signal = np.array(signal)
    activation_cloud_x = np.array(activation_cloud_x)
    segment = signal[activation_cloud_x]
                    
    # 2. Поиск максимума и минимума
    max_value = np.max(segment)
    min_value = np.min(segment)
    
    # 3. Определение наиболее выраженного экстремума
    if abs(max_value) > abs(min_value):
        extremum_index_in_segment = np.argmax(segment)  # Индекс максимума в отрезке
    else:
        extremum_index_in_segment = np.argmin(segment)  # Индекс минимума в отрезке
    
    # 4. Определение координаты экстремума
    extremum_index_in_signal = activation_cloud_x[extremum_index_in_segment]
    
    return extremum_index_in_signal


def get_delineation_from_activation_by_extremum_signal(threshold, activations, signal):
    # TODO
    activation_clouds = find_activation_clouds(threshold, activations)

    delin_coords = []
    delin_weights = []

    for cloud_x, cloud_y in activation_clouds:
        extr = find_extremum_coord(signal, cloud_x)
        delin_coords.append(extr)
        delin_weights.append(max(cloud_y))  # Вес — максимальное значение активации в облаке

    return np.array(delin_coords), np.array(delin_weights)
    
    
def get_delineation_from_activation_by_mean(threshold, activations):
    # TODO
    activation_clouds = find_activation_clouds(threshold, activations)

    delin_coords = []
    delin_weights = []

    for cloud_x, cloud_y in activation_clouds:
        probabilities_norm = cloud_y / np.sum(cloud_y)
        math_expectation = round(np.sum(cloud_x * probabilities_norm))
        delin_coords.append(math_expectation)
        delin_weights.append(max(cloud_y))  # Вес — максимальное значение активации в облаке

    return np.array(delin_coords), np.array(delin_weights)

def get_delineation_from_activation_by_max(threshold, activations):
    # TODO
    activation_clouds = find_activation_clouds(threshold, activations)

    delin_coords = []
    delin_weights = []

    for cloud_x, cloud_y in activation_clouds:
        max_point_y = max(cloud_y)
        max_point_x = cloud_x[cloud_y.index(max_point_y)]
        delin_coords.append(max_point_x)
        delin_weights.append(max_point_y)

    return np.array(delin_coords), np.array(delin_weights)