class Scenario:
    def __init__(self, scenario_entries_list):
        self.scenario_entries_list = scenario_entries_list
        self.scenario_steps = []

    def add_step(self, indexes_list):
        self.scenario_steps.append(indexes_list)

    def __len__(self):
        return len(self.scenario_steps)

    def get_entres_list_of_ith_step(self, i):
        entryes_list= []
        for index in self.scenario_steps[i]:
            entryes_list.append(self.scenario_entries_list[index])
        return entryes_list
