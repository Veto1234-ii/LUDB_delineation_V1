from .one_CNN_activations_to_delineation import get_delineation_from_activation_by_mean, get_delineation_from_activation_by_max

    
def get_democracy_delineation_by_mean(threshold, activations):
        
    delineation = get_delineation_from_activation_by_mean(threshold, activations)
    
    return delineation

def get_democracy_delineation_by_max(threshold, activations):
        
    delineation = get_delineation_from_activation_by_max(threshold, activations)

    return delineation