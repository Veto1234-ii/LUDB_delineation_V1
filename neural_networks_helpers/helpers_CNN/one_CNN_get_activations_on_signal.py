import numpy as np


def get_activations_of_CNN_on_signal(trained_CNN, one_lead_signal):
    # TODO
    signal_win_len = trained_CNN.get_win_len()
    
    activations = np.zeros(len(one_lead_signal))
    
    for p in range(0, len(one_lead_signal) - signal_win_len):
        
        # binVote = trained_CNN.apply(one_lead_signal[p:p + signal_win_len].unsqueeze(0))
        binVote = trained_CNN.apply(one_lead_signal[p:p + signal_win_len])

        
        binVote = binVote.squeeze(0).detach().numpy()
        
        activations[p + signal_win_len//2 - 1] = binVote
    
    return activations