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

dot_radius_px = 0.1 * fig_size_y * ax_size_y


def tcbase(df, ax):

    ax.plot(df.ord, df.value, linewidth=2.0, zorder=1)
    plt.xticks(df.ord, df.id.tolist(), rotation=-90)
    ax.tick_params(labelsize=6)
    ax.set_xlim([min(df.ord) - 0.5, max(df.ord) + 0.5])
    ax.set_ylim([0, 4])

    ax_width = max(df.ord) - min(df.ord) + 1

    for x, y in zip(df.ord, df.value):
        ax.add_patch(
            patches.Ellipse(
                (x, y),
                dot_radius_px,
                dot_radius_px / ((fig_size_y / fig_size_x) *
                              (ax_size_y / ax_size_x) / (4 / ax_width)),
                clip_on=False,
                zorder=200,
                color='r'
            )
        )
        ax.add_patch(
            patches.Ellipse(
                (x, y),
                dot_radius_px,
                dot_radius_px / ((fig_size_y / fig_size_x) *
                              (ax_size_y / ax_size_x) / (4 / ax_width)),
                clip_on=False,
                fill=False,
                zorder=200,
                linewidth=2,
            )
        )


def get_sep(df, group_id):
    gn = df[group_id].tolist()
    current = gn[0]
    sep = [0]
    for idx, n in enumerate(gn[1:]):
        if current != n:
            sep.append(idx + 1)
            current = n

    sep.append(idx + 2)
    return sep


def add_group_sep(df, ax, group_id, level, mode='line'):
    sep = get_sep(df, group_id)
    if mode == 'line':
        for s in sep:
            codes = [Path.MOVETO, Path.LINETO]
            vertices = [(s, (-0.72 - 0.2 * level) / (fig_size_y * ax_size_y / 4)),
                        (s, (-0.88 - 0.2 * level) / (fig_size_y * ax_size_y / 4))]
            vertices = np.array(vertices, float)
            mypath = Path(vertices, codes)
            ax.add_patch(
                patches.PathPatch(
                    mypath,
                    clip_on=False,
                    zorder=200,
                    linewidth=0.75,
                )
            )
    elif mode == 'box':
        for s1, s2 in zip(sep[1:], sep[:-1]):
            ax.add_patch(
                patches.Rectangle(
                    (s1, (-0.88 - 0.2 * level) / (fig_size_y * ax_size_y / 4)),
                    s2 - s1,
                    0.16 / (fig_size_y * ax_size_y / 4),
                    fill=False,
                    clip_on=False,
                    zorder=200,
                    linewidth=0.75,
                )
            )

    return


def readdata(fn):

    return pd.read_excel(fn, sheet_name='Sheet1')


def main(dic_arg):

    df = readdata(dic_arg['input'][0])

    if (dic_arg['group']) != None:
        df.sort_values(by=dic_arg['group'], inplace=True)

    df['ord'] = np.arange(0.5, len(df.id.tolist()) + 0.5)

    plt.clf()
    plt.close()

    fig = plt.figure(figsize=[fig_size_x, fig_size_y], dpi=fig_dpi)
    ax = fig.add_axes([ax_offset_x, ax_offset_y, ax_size_x, ax_size_y])

    tcbase(df, ax)

    if (dic_arg['group']) != None:
        for level, group_id in enumerate(dic_arg['group']):
            add_group_sep(df, ax, group_id, level,mode='box')

    plt.savefig(dic_arg['output'][0])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Produce pivot charts from data')
    parser.add_argument('-i', '--input', metavar='INPUT',
                        nargs=1, help="excel file", required=True)
    parser.add_argument('-o', '--output', metavar='OUTPUT',
                        nargs=1, help="output file path", required=True)
    parser.add_argument('-g', '--group', metavar='GROUPS',
                        nargs='+', help="group ids")

    args = parser.parse_args()

    main(vars(args))
