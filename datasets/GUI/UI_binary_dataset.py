from datasets.binary_datasets import BinaryDataset
from datasets.GUI.ecg_viewer import ECGViewer  # подключение интерфейса
import tkinter as tk

class UIBinaryDataset:
    def __init__(self, binary_dataset):
        train_data = binary_dataset.get_train()

        signals = train_data["signals"]  # проверить, как хранятся сигналы
        answers = train_data["answers"]  # метки (0 или 1)

        # запуск GUI
        root = tk.Tk()
        app = ECGViewer(root, signals, answers)
        root.mainloop()