from settings import FREQUENCY

import matplotlib.pyplot as plt

def plot_one_lead_delineation_on_ax(ax, delineation_t, Y_max, color=None, legend="пики QRS"):
    # TODO legend одна на все палочки (чтоб не каждой своя подпись, а одна подпись всего)
    # TODO if color is None: color = случайный цвет
    for x in delineation_t:
        ax.axvline(x=x, ymax=Y_max, ymin=0, color=color, linewidth=0.5)