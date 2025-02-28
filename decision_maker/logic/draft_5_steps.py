from neural_networks_models import CNN
from neural_networks_helpers import get_appliable
from settings import LEADS_NAMES, PATH_TO_LUDB

class LogicDraft5:
    def __init__(self, leads_signal, leads_names):
        self.leads_signal = leads_signal
        self.leads_names = leads_names

        # загружаем сети по их именам из папки SAVED_NETS
        self.apliable_QRS_peak_net = get_appliable('сеть1')
        self.apliable_tratrata_net = get_appliable('сеть2trtrt')

    def step_0(self):
        # TODO активации qrs-пик в i, ii

       pass

    def step_1(self):
        # TODO разметка qrs -peak в i и ii

      pass

    def step_2(self):
        # TODO активации пик p в i, ii

       pass

    def step_3(self):
        # TODO активации пик t в i, ii

       pass

    def step_4(self):
        # TODO разметка пик t, p в i, ii

        pass

if __name__ == "__main__":

    from datasets.LUDB_utils import get_LUDB_data, get_some_test_patient_id, get_signals_by_id_several_leads_mkV

    LUDB_data= get_LUDB_data()
    patient_id=  get_some_test_patient_id
    signals, leads_names = get_signals_by_id_several_leads_mkV(patient_id, LUDB_data, leads_names_list=[LEADS_NAMES.i, LEADS_NAMES.ii])

    logic_draft = LogicDraft5(signals, leads_names=leads_names)

