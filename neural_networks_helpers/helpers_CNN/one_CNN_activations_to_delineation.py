import numpy as np

def get_delineation_from_activation_by_mean(threshold, activations):
    # TODO
    delineation = []
    delineation_activations = []
    
    activation_cloud_y = []
    activation_cloud_x = []
    
    
    for i in range(len(activations)-1):
        
        if activations[i] > threshold:
            
            activation_cloud_y.append(activations[i])
            activation_cloud_x.append(i)
            
            if activations[i+1] < threshold:
                
                probabilities_norm = activation_cloud_y / np.sum(activation_cloud_y)

                math_expectation = round(np.sum(activation_cloud_x * probabilities_norm))                   
                
                delineation.append(math_expectation)
                delineation_activations.append(activations[math_expectation])
                
                
                activation_cloud_x.clear()
                activation_cloud_y.clear()
                
    # np.array(coords), np.array(activations)
    return np.array(delineation)

def get_delineation_from_activation_by_max(threshold, activations):
    # TODO
    
    delineation = []
    delineation_activations = []
    
    activation_cloud_x = []
    activation_cloud_y = []
    
    
    for i in range(len(activations)-1):
        
        if activations[i] > threshold:
            activation_cloud_y.append(activations[i])
            activation_cloud_x.append(i)
            
            if activations[i+1] < threshold:
                max_point_y = max(activation_cloud_y)
                max_point_x = activation_cloud_x[activation_cloud_y.index(max_point_y)]
                
                delineation.append(max_point_x)
                delineation_activations.append(max_point_y)
                
                activation_cloud_x.clear()
                activation_cloud_y.clear()
                
    return np.array(delineation)

