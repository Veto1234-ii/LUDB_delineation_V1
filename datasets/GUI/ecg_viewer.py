import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from visualisation_utils import plot_lead_signal_to_ax

class ECGViewer:
    def __init__(self, root, signals, answers):
        self.root = root
        self.signals = signals
        self.answers = answers
        self.index = 0  # начальный индекс сигнала

        self.root.title("ECG Viewer")

        # некоторый фрейм для графика
        self.frame = ttk.Frame(root)
        self.frame.pack()

        # фигура и ось для графика
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack()

        # поле для ответа
        self.answer_entry = ttk.Entry(root, font=("Arial", 14), justify="center")
        self.answer_entry.pack(pady=10)

        # кнопки :)
        self.button_frame = ttk.Frame(root)
        self.button_frame.pack()

        self.prev_button = ttk.Button(self.button_frame, text="Назад", command=self.prev_signal)
        self.prev_button.grid(row=0, column=0, padx=5)

        self.next_button = ttk.Button(self.button_frame, text="Вперед", command=self.next_signal)
        self.next_button.grid(row=0, column=1, padx=5)

        # отображение первого сигнала и т.д.
        self.update_plot()

    def update_plot(self):
        """обновление графика и текстового поля с ответом"""
        self.ax.clear()
        plot_lead_signal_to_ax(self.signals[self.index], self.ax)  # МЕТОД из visualisation_utils
        self.ax.set_title(f"Сигнал {self.index + 1}/{len(self.signals)}")
        self.canvas.draw()

        # обновление значения в текстовом поле
        self.answer_entry.delete(0, tk.END)
        self.answer_entry.insert(0, str(self.answers[self.index]))

    def next_signal(self):
        """переключение на следующий сигнал"""
        if self.index < len(self.signals) - 1:
            self.index += 1
            self.update_plot()

    def prev_signal(self):
        """переключение на предыдущий сигнал"""
        if self.index > 0:
            self.index -= 1
            self.update_plot()


if __name__ == "__main__":
    root = tk.Tk()

    # генерация тестовых данных (замена на реальные)
    num_signals = 15
    signal_length = 500  # 1 секунда при 500 Гц
    test_signals = [np.sin(np.linspace(0, 2 * np.pi, signal_length)) for _ in range(num_signals)]
    test_answers = np.random.randint(0, 2, num_signals)

    app = ECGViewer(root, test_signals, test_answers)
    root.mainloop()