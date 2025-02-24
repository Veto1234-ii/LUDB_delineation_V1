from settings import FREQUENCY

import matplotlib.pyplot as plt

def plot_one_lead_delineation_on_ax(ax, delineation, Y_max, color=None, legend=""):
    # TODO legend одна на все палочки (чтоб не каждой своя подпись, а одна подпись всего)
    # TODO if color is None: color = случайный цвет
    for x in delineation:
        ax.axvline(x=x/FREQUENCY, ymax=Y_max, ymin=0, color=color, linewidth=0.5)