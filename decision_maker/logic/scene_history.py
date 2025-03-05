class SceneHistory:
    def __init__(self):
        self.visibles_groups = []
        self.invisibles_groups = []

    def add_entry(self, visibles=[], invisibles=[]):
        if not(len(visibles)==0 and len(invisibles) ==0):
            self.visibles_groups.append(visibles)
            self.invisibles_groups.append(invisibles)

    def _update_visibles(self, prev_visibles, new_visibles, new_invisibles):
        current_visibles = set(prev_visibles + new_visibles)
        new_unvisibles = set(new_invisibles)
        updated_visibles = current_visibles - new_unvisibles
        return list(updated_visibles)

    def get_ids_for_step_i(self, i):
        current_visibles = []
        for i in range( i+1):
            new_invisibles = self.invisibles_groups[i]
            new_visibles = self.visibles_groups[i]
            current_visibles = self._update_visibles(prev_visibles=current_visibles,
                                                     new_visibles=new_visibles,
                                                     new_invisibles=new_invisibles)
        return current_visibles

    def __len__(self):
        return len(self.visibles_groups)


if __name__ == "__main__":
    scene_history = SceneHistory()
    scene_history.add_entry(visibles=[1,2,3])
    scene_history.add_entry(invisibles=[2])
    scene_history.add_entry(visibles=[2, 4, 5], invisibles=[1])

    for i in range(len(scene_history)):
        print(scene_history.get_ids_for_step_i(i))