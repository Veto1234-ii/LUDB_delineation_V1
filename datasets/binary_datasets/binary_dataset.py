import numpy as np
class BinaryDataset:
    def __init__(self, name, signals_train, signals_test, labels_train, labels_test):
        self.name = name

        self.signals_train = signals_train
        self.signals_test = signals_test
        self.labels_train = labels_train
        self.labels_test = labels_test
        # # numpy массивы с обучающими данными
        # self.dataset_test = None
        # self.dataset_train = None

    def get_test(self):
        # return self.dataset_test
        return self.signals_test, self.labels_test

    def get_train(self):
        # return self.dataset_train
        return self.signals_train, self.labels_train

    def get_name(self):
        return self.name


    def shuffle(self):
        # Перемешиваем обучающие данные
        indices_train = np.random.permutation(len(self.signals_train))
        self.signals_train = self.signals_train[indices_train]
        self.labels_train = self.labels_train[indices_train]

        # Перемешиваем тестовые данные
        indices_test = np.random.permutation(len(self.signals_test))
        self.signals_test = self.signals_test[indices_test]
        self.labels_test = [self.labels_test[i] for i in indices_test]