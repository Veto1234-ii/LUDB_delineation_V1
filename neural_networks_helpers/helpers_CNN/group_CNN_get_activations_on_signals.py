from .one_CNN_get_activations_on_signal import get_activations_of_CNN_on_signal
import numpy as np


def get_activations_of_group_CNN(list_CNNs, list_signals):
    avg = []

    for i in range(len(list_signals)):
        activations = get_activations_of_CNN_on_signal(list_CNNs[i], list_signals[i])
        avg.append(activations)

    result_activations = np.mean(avg, axis=0)
    
    return result_activations

