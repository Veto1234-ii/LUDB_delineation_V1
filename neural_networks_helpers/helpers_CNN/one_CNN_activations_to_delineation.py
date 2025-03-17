import numpy as np

def get_delineation_from_activation_by_mean(threshold, activations):
    # TODO
    delin_coords = []
    delin_weights = []
    
    activation_cloud_y = []
    activation_cloud_x = []
    
    
    for i in range(len(activations)-1):
        
        if activations[i] > threshold:
            
            activation_cloud_y.append(activations[i])
            activation_cloud_x.append(i)
            
            if activations[i+1] < threshold:
                
                probabilities_norm = activation_cloud_y / np.sum(activation_cloud_y)

                math_expectation = round(np.sum(activation_cloud_x * probabilities_norm))                   
                
                delin_coords.append(math_expectation)
                
                # максимальное значение активации внутри этого облака
                delin_weights.append(max(activation_cloud_y))
                
                
                activation_cloud_x.clear()
                activation_cloud_y.clear()
                
    return np.array(delin_coords), np.array(delin_weights)

def get_delineation_from_activation_by_max(threshold, activations):
    # TODO
    
    delin_coords = []
    delin_weights = []
    
    activation_cloud_x = []
    activation_cloud_y = []
    
    
    for i in range(len(activations)-1):
        
        if activations[i] > threshold:
            activation_cloud_y.append(activations[i])
            activation_cloud_x.append(i)
            
            if activations[i+1] < threshold:
                max_point_y = max(activation_cloud_y)
                max_point_x = activation_cloud_x[activation_cloud_y.index(max_point_y)]
                
                delin_coords.append(max_point_x)
                delin_weights.append(max_point_y)
                
                activation_cloud_x.clear()
                activation_cloud_y.clear()
                
    return np.array(delin_coords), np.array(delin_weights)

