from neural_networks_models import CNN
from neural_networks_helpers import get_appliable
from settings import LEADS_NAMES, PATH_TO_LUDB

from neural_networks_helpers.helpers_CNN.one_CNN_get_activations_on_signal import get_activations_of_CNN_on_signal
from neural_networks_helpers.helpers_CNN.one_CNN_activations_to_delineation import get_delineation_from_activation_by_mean





class LogicDraft5:
    
    def __init__(self, leads_signal, leads_names):
        
        self.leads_signal = leads_signal
        self.leads_names = leads_names

        # загружаем сети по их именам из папки SAVED_NETS
        
        # QRS I, II
        self.cnn_i_qrs = get_appliable('cnn_i_qrs')
        self.cnn_ii_qrs = get_appliable('cnn_ii_qrs')
        
        # P I, II
        self.cnn_i_p = get_appliable('cnn_i_p')
        self.cnn_ii_p = get_appliable('cnn_ii_p')
        
        # T I, II
        self.cnn_i_t = get_appliable('cnn_i_t')
        self.cnn_ii_t = get_appliable('cnn_ii_t')

        
    def step_0(self):
        # TODO активации qrs-пик в i, ii
        
        activations_qrs_i = get_activations_of_CNN_on_signal(self.cnn_i_qrs, self.leads_signal[0])
        activations_qrs_ii = get_activations_of_CNN_on_signal(self.cnn_ii_qrs, self.leads_signal[1])


    def step_1(self):
        # TODO разметка qrs -peak в i и ii
        activations_qrs_i = get_activations_of_CNN_on_signal(self.cnn_i_qrs, self.leads_signal[0])
        activations_qrs_ii = get_activations_of_CNN_on_signal(self.cnn_ii_qrs, self.leads_signal[1])
        
        threshold = 0.9
        delineation_qrs_i = get_delineation_from_activation_by_mean(threshold, activations_qrs_i)
        delineation_qrs_ii = get_delineation_from_activation_by_mean(threshold, activations_qrs_ii)

        print(delineation_qrs_i)

    def step_2(self):
        # TODO активации пик p в i, ii

       activations_p_i = get_activations_of_CNN_on_signal(self.cnn_i_p, self.leads_signal[0])
       activations_p_ii = get_activations_of_CNN_on_signal(self.cnn_ii_p, self.leads_signal[1])

    def step_3(self):
        # TODO активации пик t в i, ii
        
        activations_t_i = get_activations_of_CNN_on_signal(self.cnn_i_t, self.leads_signal[0])
        activations_t_ii = get_activations_of_CNN_on_signal(self.cnn_ii_t, self.leads_signal[1])

       

    def step_4(self):
        # TODO разметка пик t, p в i, ii
        activations_p_i = get_activations_of_CNN_on_signal(self.cnn_i_p, self.leads_signal[0])
        activations_p_ii = get_activations_of_CNN_on_signal(self.cnn_ii_p, self.leads_signal[1])
        
        threshold = 0.8
        delineation_p_i = get_delineation_from_activation_by_mean(threshold, activations_p_i)
        delineation_p_ii = get_delineation_from_activation_by_mean(threshold, activations_p_ii)


        activations_t_i = get_activations_of_CNN_on_signal(self.cnn_i_t, self.leads_signal[0])
        activations_t_ii = get_activations_of_CNN_on_signal(self.cnn_ii_t, self.leads_signal[1])
        
        threshold = 0.9
        delineation_t_i = get_delineation_from_activation_by_mean(threshold, activations_t_i)
        delineation_t_ii = get_delineation_from_activation_by_mean(threshold, activations_t_ii)
        
        print(delineation_p_i)


        

if __name__ == "__main__":

    from datasets.LUDB_utils import get_LUDB_data, get_some_test_patient_id, get_signals_by_id_several_leads_mkV

    LUDB_data= get_LUDB_data()
    patient_id=  get_some_test_patient_id()
    signals, leads_names = get_signals_by_id_several_leads_mkV(patient_id, LUDB_data, leads_names_list=[LEADS_NAMES.i, LEADS_NAMES.ii])

    logic_draft = LogicDraft5(signals, leads_names=leads_names)
    
    logic_draft.step_0()
    logic_draft.step_1()
    logic_draft.step_2()
    logic_draft.step_3()
    logic_draft.step_4()

    

