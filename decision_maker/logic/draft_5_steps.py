from decision_maker.GUI import Scenario, ScenarioEntry, GUI_ToShowScenario
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


        self.scenario_entryes_list = []

        self.step_0()
        self.step_1()
        self.step_2()
        self.step_3()
        self.step_4()


    def create_scenario(self):
        scenario = Scenario(scenario_entries_list=self.scenario_entryes_list)
        scenario.add_step(indexes_list=[0, 1])
        scenario.add_step(indexes_list=[0, 1, 2, 3])
        scenario.add_step(indexes_list=[ 2,3,4,5])
        scenario.add_step(indexes_list=[2,3,4,5, 6, 7])
        scenario.add_step(indexes_list=[ 2,3,hjhjkh])
        return scenario

    def step_0(self):
        # TODO активации qrs в i, ii

        # Резульаты заносим в запись для будущей отрисовки в GUI
        scneario_entry_i= ScenarioEntry(bla bla bla)
        self.scenario_entryes_list.append(scneario_entry)

        scneario_entry_ii = ScenarioEntry()
        self.scenario_entryes_list.append(scneario_entry)

    def step_1(self):
        # TODO qrs -peak в i и ii

        # Резульаты заносим в запись для будущей отрисовки в GUI
        scneario_entry1 = ScenarioEntry(bla bla bla)
        self.scenario_entryes_list.append(scneario_entry)

        scneario_entry1 = ScenarioEntry(ff)
        self.scenario_entryes_list.append(scneario_entry)

    def step_2(self):
        # TODO активации пик p в i, ii

        # Резульаты заносим в запись для будущей отрисовки в GUI
        scneario_entry1 = ScenarioEntry()
        self.scenario_entryes_list.append(scneario_entry)

        scneario_entry1 = ScenarioEntry(ff)
        self.scenario_entryes_list.append(scneario_entry)

    def step_3(self):
        # TODO активации пик t в i, ii

        # Резульаты заносим в запись для будущей отрисовки в GUI
        scneario_entry1 = ScenarioEntry()
        self.scenario_entryes_list.append(scneario_entry)

        scneario_entry1 = ScenarioEntry(ff)
        self.scenario_entryes_list.append(scneario_entry)

    def step_4(self):
        # TODO разметка пик t, p в i, ii

        # Резульаты заносим в запись для будущей отрисовки в GUI
        scneario_entry_i_t = ScenarioEntry()
        self.scenario_entryes_list.append(scneario_entry)

        scneario_entry_ii_t = ScenarioEntry(ff)
        self.scenario_entryes_list.append(scneario_entry)

        scneario_entry_i_p = ScenarioEntry()
        self.scenario_entryes_list.append(scneario_entry)

        scneario_entry_ii_p = ScenarioEntry(ff)
        self.scenario_entryes_list.append(scneario_entry)

if __name__ == "__main__":

    from decision_maker.GUI import GUI_ToShowScenario
    from datasets.LUDB_utils import get_LUDB_data, get_some_test_patient_id, get_signals_by_id_several_leads_mkV

    LUDB_data= get_LUDB_data()
    patient_id=  get_some_test_patient_id
    signals, leads_names = get_signals_by_id_several_leads_mkV(patient_id, LUDB_data, leads_names_list=[LEADS_NAMES.i, LEADS_NAMES.ii])

    logic_draft = LogicDraft5(signals, leads_names=leads_names)
    scenario = logic_draft.create_scenario()

    #gui = GUI_ToShowScenario(signals, leads_names, scenario)

