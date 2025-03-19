import numpy as np

class SegmentationDataset:
    def __init__(self, name, signals_train, signals_test, masks_train, masks_test):
        self.name = name
        self.signals_train = signals_train
        self.signals_test = signals_test
        self.masks_train = masks_train
        self.masks_test = masks_test

    def get_test(self):
        return self.signals_test, self.masks_test

    def get_train(self):
        return self.signals_train, self.masks_train

    def get_name(self):
        return self.name

    def shuffle(self):
        # Перемешиваем обучающие данные
        indices_train = np.random.permutation(len(self.signals_train))
        self.signals_train = self.signals_train[indices_train]
        self.masks_train = self.masks_train[indices_train]

        # Перемешиваем тестовые данные
        indices_test = np.random.permutation(len(self.signals_test))
        self.signals_test = self.signals_test[indices_test]
        self.masks_test = self.masks_test[indices_test]