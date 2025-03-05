from visualisation_utils import plot_lead_signal_to_ax

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class UI:
    def __init__(self, signals, leads_names):

        self.signals = signals
        self.leads_names = leads_names


        self.root = tk.Tk()
        self.root.state('zoomed')  # Запускаем окно в полноэкранном режиме
        self.root.title("Просмотрщик одной ЭКГ")

        # Создаем верхний и нижний фреймы
        self.top_frame = ttk.Frame(self.root)
        self.bottom_frame = ttk.Frame(self.root)

        self.top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

        # Создаем графики на верхнем фрейме
        self.create_plots(self.top_frame)

        # Создаем нижний фрейм c текстом
        entry1 = tk.Text(self.bottom_frame, wrap=tk.WORD)
        entry1.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        entry1.insert(tk.END, "ЛОГ: " )

        # Привязываем обработчик события изменения размера окна
        self.root.bind("<Configure>", self.update_frames)
        self.update_frames()

        # Запускаем главный цикл
        self.root.mainloop()

    def update_frames(self, event=None):
        self.top_frame.config(height=self.root.winfo_height() * 0.8)
        self.bottom_frame.config(height=self.root.winfo_height() * 0.2)

    def get_MAX_MIN_Y(self):
        mins = []
        maxs = []
        for signal in self.signals:
            mins.append(np.min(signal))
            maxs.append(np.max(signal))
        overall_min = min(mins)
        overall_max = max(maxs)
        return overall_max, overall_min

    def create_plots(self, parent):
        n = len(self.leads_names)

        # Вычисляем высоту каждого рисунка как долю от высоты верхнего фрейма
        plot_height = 2.0 / n  # Высота каждого графика (в дюймах)
        Y_max, Y_min = self.get_MAX_MIN_Y()

        for i in range(n):
            signal = self.signals[i]
            name = self.leads_names[i]

            # создаем рисунок
            fig = plt.Figure(figsize=(6, plot_height))
            ax = fig.add_subplot(111)

            # Рисуем на нем сигнал
            plot_lead_signal_to_ax(ax=ax, signal_mV=signal, Y_max=Y_max,
                     Y_min=Y_min)

            # Создаем фрейм для рисунка и подписи к нему
            lead_frame = ttk.Frame(parent)
            lead_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Вставляем рисуок к фрейм
            canvas = FigureCanvasTkAgg(fig, master=lead_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Вставляем подпись (имя отведения) справа от рисунка
            label = tk.Label(lead_frame, text=name, font=("Arial", 14))
            label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, pady=10)

if __name__ == "__main__":
    # какие отведения хотим показать
    from settings import LEADS_NAMES
    from datasets.LUDB_utils import get_signals_by_id_several_leads_mV, get_LUDB_data, get_some_test_patient_id
    leads_names =[LEADS_NAMES.i, LEADS_NAMES.ii, LEADS_NAMES.iii]

    LUDB_data = get_LUDB_data()
    patient_id = get_some_test_patient_id()
    signals_list, leads_names_list = get_signals_by_id_several_leads_mV(patient_id=patient_id, LUDB_data=LUDB_data,leads_names_list=leads_names)

    ui = UI(leads_names=leads_names_list, signals=signals_list)