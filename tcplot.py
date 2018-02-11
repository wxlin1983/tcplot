import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import sys


def tc(arglist):

    fig_size_x = 10
    fig_size_y = 2.5
    fig_dpi = 300

    ax_offset_x = 0.05
    ax_offset_y = 0.45
    ax_size_x = 0.925
    ax_size_y = 0.5

    df = pd.read_excel(arglist[0], sheet_name='Sheet1')

    matplotlib.rc('lines', linewidth=2, color='r')

    plt.clf()
    plt.close()

    fig = plt.figure(figsize=[fig_size_x, fig_size_y], dpi=fig_dpi)
    ax = fig.add_axes([ax_offset_x, ax_offset_y, ax_size_x, ax_size_y])

    od = list(range(1, len(df.id.tolist()) + 1))
    ax.plot(od, df.value, linewidth=2.0, zorder=1)
    plt.xticks(od, df.id.tolist(), rotation=-90)
    ax.tick_params(labelsize=6)
    ax.set_xlim([0.5, 7.5])
    ax.set_ylim([0, 4])

    dot_radius = 0.1 * fig_size_y * ax_size_y
    for x, y in zip(od, df.value):
        ax.add_patch(
            patches.Ellipse(
                (x, y),   # (x,y)
                dot_radius,          # height
                dot_radius / ((fig_size_y / fig_size_x) * \
                              (ax_size_y / ax_size_x) / (4 / 7)),
                clip_on=False,
                zorder=200,
                color='r'
            )
        )
        ax.add_patch(
            patches.Ellipse(
                (x, y),   # (x,y)
                dot_radius,          # height
                dot_radius / ((fig_size_y / fig_size_x) * \
                              (ax_size_y / ax_size_x) / (4 / 7)),
                clip_on=False,
                fill=False,
                zorder=200,
                linewidth=2,
            )
        )

    plt.savefig(arglist[1])


if __name__ == '__main__':

    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    tc(sys.argv[1:])
