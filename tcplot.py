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
ax_size_x_ratio = 0.925
ax_size_y_ratio = 0.5

fig_size_x_inch = 10
fig_size_y_inch = 2.5
dot_radius_inch = 0.1

fig_dpi = 300


def tcbase(df, ax, dic_plot_para):

    xlim_min = dic_plot_para['xlim_min']
    xlim_max = dic_plot_para['xlim_max']
    ylim_min = dic_plot_para['ylim_min']
    ylim_max = dic_plot_para['ylim_max']

    ax.plot(df.ORD, df.value, linewidth=2.0, zorder=1)
    plt.xticks(df.ORD, df.id.tolist(), rotation=-90)
    ax.tick_params(labelsize=6)

    ax.set_xlim([xlim_min, xlim_max])
    ax.set_ylim([ylim_min, ylim_max])

    ax_width = max(df.ORD) - min(df.ORD) + 1

    dot_radius_x = (dot_radius_inch / fig_size_y_inch *
                    ax_size_y_ratio) * ax_width
    for x, y in zip(df.ORD, df.value):
        ax.add_patch(
            patches.Ellipse(
                (x, y),
                dot_radius_x,
                dot_radius_x / ((fig_size_y_inch / fig_size_x_inch) *
                                (ax_size_y_ratio / ax_size_x_ratio) / (4 / ax_width)),
                clip_on=False,
                zorder=200,
                color='r'
            )
        )
        ax.add_patch(
            patches.Ellipse(
                (x, y),
                dot_radius_x,
                dot_radius_x / ((fig_size_y_inch / fig_size_x_inch) *
                                (ax_size_y_ratio / ax_size_x_ratio) / (4 / ax_width)),
                clip_on=False,
                fill=False,
                zorder=200,
                linewidth=2,
            )
        )


def add_group_sep(df, ax, group_id, level, mode='line'):
    sep = get_sep(df, group_id)
    if mode == 'line':
        for s in sep:
            codes = [Path.MOVETO, Path.LINETO]
            vertices = [(s, (-0.72 - 0.2 * level) / (fig_size_y_inch * ax_size_y_ratio / 4)),
                        (s, (-0.88 - 0.2 * level) / (fig_size_y_inch * ax_size_y_ratio / 4))]
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
                    (s1, (-0.88 - 0.2 * level) /
                     (fig_size_y_inch * ax_size_y_ratio / 4)),
                    s2 - s1,
                    0.16 / (fig_size_y_inch * ax_size_y_ratio / 4),
                    fill=False,
                    clip_on=False,
                    zorder=200,
                    linewidth=0.75,
                )
            )

    return


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


def readdata(fn):

    try:
        out = pd.read_excel(fn, sheet_name='Sheet1')
    except:
        out = pd.read_csv(fn)

    return out


def main(dic_arg):

    df = readdata(dic_arg['input'][0])

    if (dic_arg['group']) != None:
        df.sort_values(by=dic_arg['group'], inplace=True)

    df['ORD'] = np.arange(0.5, len(df.id.tolist()) + 0.5)

    dic_plot_para = dict()

    dic_plot_para['xlim_min'] = 0
    dic_plot_para['xlim_max'] = len(df.id.tolist())
    dic_plot_para['ylim_min'] = 0
    if dic_arg['ymax'] != None:
        dic_plot_para['ylim_max'] = dic_arg['ymax'][0]
    else:
        dic_plot_para['ylim_max'] = max(df.value)

    plt.clf()
    plt.close()

    fig = plt.figure(figsize=[fig_size_x_inch, fig_size_y_inch], dpi=fig_dpi)
    ax = fig.add_axes(
        [ax_offset_x, ax_offset_y, ax_size_x_ratio, ax_size_y_ratio])

    tcbase(df, ax, dic_plot_para)

    if (dic_arg['group']) != None:
        for level, group_id in enumerate(dic_arg['group']):
            add_group_sep(df, ax, group_id, level, mode='box')

    plt.savefig(dic_arg['output'][0])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Plot pivot charts from data')

    parser.add_argument('input', metavar='INPUT',
                        nargs=1, help="input excel or csv file")
    parser.add_argument('-o', '--output', metavar='OUTPUT',
                        nargs=1, help="output file path")
    parser.add_argument('-g', '--group', metavar='GROUPS',
                        nargs='+', help="group by...")
    parser.add_argument('-x', metavar='X',
                        nargs=1, help="data x")
    parser.add_argument('-y', metavar='Y',
                        nargs=1, help="data y")
    parser.add_argument('--ymax', metavar='YMAX',
                        nargs=1, type=int, help="data y upper limit")

    args = parser.parse_args()

    main(vars(args))
