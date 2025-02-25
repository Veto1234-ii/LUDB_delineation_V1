from plot_one_lead_delineation import plot_one_lead_delineation_on_ax

def plot_several_delins_on_ax(ax, delineations, Y_max, colors, names):
    for i in range(len(delineations)):
        plot_one_lead_delineation_on_ax(ax,
                                        delineation=delineations[i],
                                        Y_max=Y_max,
                                        color=colors[i],
                                        legend=names[i])
