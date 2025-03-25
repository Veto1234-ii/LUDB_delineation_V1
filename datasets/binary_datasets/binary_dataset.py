import numpy as np
import random


class BinaryDataset:
    def __init__(self, name, signals_train, signals_test, labels_train, labels_test, full_signals, delineation_on_full_signals):
        self.name = name
        self.signals_train = signals_train
        self.signals_test = signals_test
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.full_signals = full_signals
        self.delineation_on_full_signals = delineation_on_full_signals


    def get_test(self):
        return self.signals_test, self.labels_test

    def get_train(self):
        return self.signals_train, self.labels_train

    def get_name(self):
        return self.name


    def shuffle(self):
        # Перемешиваем обучающие данные
        indices_train = np.random.permutation(len(self.signals_train))
        self.signals_train = self.signals_train[indices_train]
        self.labels_train = self.labels_train[indices_train]
        self.full_signals = self.full_signals[indices_train]
        self.delineation_on_full_signals = self.delineation_on_full_signals[indices_train]

        # Перемешиваем тестовые данные
        indices_test = np.random.permutation(len(self.signals_test))
        self.signals_test = self.signals_test[indices_test]
        self.labels_test = [self.labels_test[i] for i in indices_test]


    def add_jitter(self, num_of_jitter_examples, jitter_range=range(50, 29, -10)):
        """

            Args:
                num_of_jitter_examples: количество примеров для добавления
                jitter_range=range(50, 29, -10): смещения от центра исходных окон

        """

        if num_of_jitter_examples <= 0:
            raise ValueError("num_of_jitter_examples должно быть > 0")
        radius = len(self.signals_train[0]) // 2
        new_signals = []
        new_full_signals = []
        new_delineation_on_full_signals = []

        positive_indices = []
        for i in range(len(self.labels_train)):
            if self.labels_train[i] == 1:
                positive_indices.append(i)

        for _ in range(num_of_jitter_examples):
            idx = random.choice(positive_indices)
            full_signal = self.full_signals[idx]
            original_center = self.delineation_on_full_signals[idx]

            offset = random.choice(jitter_range) * random.choice([-1, 1])
            new_center = original_center + offset

            if (0 <= new_center - radius) and (new_center + radius <= len(full_signal)):
                new_signals.append(full_signal[new_center - radius:new_center + radius])
                new_full_signals.append(full_signal)
                new_delineation_on_full_signals.append(new_center)

        if new_signals:
            self.signals_train = np.concatenate([self.signals_train, np.array(new_signals)])
            self.labels_train = np.concatenate([self.labels_train, np.zeros(len(new_signals), dtype=int)])
            self.full_signals = np.concatenate([self.full_signals, np.array(new_full_signals)])
            self.delineation_on_full_signals = np.concatenate([self.delineation_on_full_signals, np.array(new_delineation_on_full_signals)])

        print(f"Добавлено {len(new_signals)} примеров с джиттером")