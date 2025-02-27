from settings import FREQUENCY, DELINEATION_LINEWIDTH

import matplotlib.pyplot as plt
import random

def plot_one_lead_delineation_on_ax(ax, delineation_t, Y_max, color=None, legend=""):
    # TODO legend одна на все палочки (чтоб не каждой своя подпись, а одна подпись всего)
    if color is None:
        color = (random.random(), random.random(), random.random())
    for x in delineation_t:
        ax.axvline(x=x, ymax=Y_max, ymin=0, color=color, linewidth=DELINEATION_LINEWIDTH)