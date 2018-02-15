import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import sys
import getopt
import argparse

ax_offset_x = 0.05
ax_offset_y = 0.45
ax_size_x = 0.925
ax_size_y = 0.5

fig_size_x = 10
fig_size_y = 2.5
fig_dpi = 300


def tcbase(df, ax):

    ax.plot(df.ord, df.value, linewidth=2.0, zorder=1)
    plt.xticks(df.ord, df.id.tolist(), rotation=-90)
    ax.tick_params(labelsize=6)
    ax.set_xlim([min(df.ord) - 0.5, max(df.ord) + 0.5])
    ax.set_ylim([0, 4])

    ax_width = max(df.ord) - min(df.ord) + 1
    dot_radius = 0.1 * fig_size_y * ax_size_y
    for x, y in zip(df.ord, df.value):
        ax.add_patch(
            patches.Ellipse(
                (x, y),
                dot_radius,
                dot_radius / ((fig_size_y / fig_size_x) *
                              (ax_size_y / ax_size_x) / (4 / ax_width)),
                clip_on=False,
                zorder=200,
                color='r'
            )
        )
        ax.add_patch(
            patches.Ellipse(
                (x, y),
                dot_radius,
                dot_radius / ((fig_size_y / fig_size_x) *
                              (ax_size_y / ax_size_x) / (4 / ax_width)),
                clip_on=False,
                fill=False,
                zorder=200,
                linewidth=2,
            )
        )


def get_sep(df, group_name):
    gn = df[group_name].tolist()
    current = gn[0]
    sep = [0]
    for idx, n in enumerate(gn[1:]):
        if current != n:
            sep.append(idx + 1)
            current = n

    sep.append(idx + 2)
    return sep


def add_group_line(df, ax, group_name):
    sep = get_sep(df, group_name)
    for s in sep:
        codes = [Path.MOVETO, Path.LINETO]
        vertices = [(s, -0.7 / (fig_size_y * ax_size_y / 4)),
                    (s, -1 / (fig_size_y * ax_size_y / 4))]
        vertices = np.array(vertices, float)
        mypath = Path(vertices, codes)
        ax.add_patch(
            patches.PathPatch(
                mypath,
                clip_on=False,
                zorder=200,
                linewidth=0.5,
            )
        )
    return


def readdata(fn):

    return pd.read_excel(fn, sheet_name='Sheet1')


def main(dic_arg):

    df = readdata(dic_arg['input'][0])

    df.sort_values(by='t1', inplace=True)
    df['ord'] = np.arange(0.5, len(df.id.tolist()) + 0.5)

    plt.clf()
    plt.close()

    fig = plt.figure(figsize=[fig_size_x, fig_size_y], dpi=fig_dpi)
    ax = fig.add_axes([ax_offset_x, ax_offset_y, ax_size_x, ax_size_y])

    tcbase(df, ax)
    add_group_line(df, ax, 't1')

    plt.savefig(dic_arg['output'][0])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Produce Pivot Chart')
    parser.add_argument('-i', '--input', metavar='INPUT',
                        nargs=1, help="excel file", required=True)
    parser.add_argument('-o', '--output', metavar='OUTPUT',
                        nargs=1, help="output file path", required=True)

    args = parser.parse_args()

    main(vars(args))
