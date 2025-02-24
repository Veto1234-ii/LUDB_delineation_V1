
class BinaryDataset:
    def __init__(self, name):
        self.name = name

        # numpy массивы с обучающими данными
        self.dataset_test = None
        self.dataset_train = None

    def get_test(self):
        return self.dataset_test

    def get_train(self):
        return self.dataset_train

    def get_name(self):
        return self.name


