import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import argparse

ax_offset_x_ratio = 0.05
ax_offset_y_ratio = 0.45
ax_size_x_ratio = 0.925
ax_size_y_ratio = 0.5

fig_size_x_in = 10
fig_size_y_in = 2.5
dot_radius_in = 0.1

igroup_indi_offset_y_in = 0.6
igroup_indi_height_y_in = 0.2
igroup_indi_sep_y_in = 0.02

fig_dpi = 300


def tcbase(df, ax, para):

    lim_x_min = para['lim_x_min']
    lim_x_max = para['lim_x_max']
    lim_y_min = para['lim_y_min']
    lim_y_max = para['lim_y_max']

    ax_height = para['lim_y_max'] - para['lim_y_min']
    ax_width = para['lim_x_max'] - para['lim_x_min']

    ax.plot(df.ORD, df.value, 'k', linewidth=2.0, zorder=1)
    plt.xticks(df.ORD, df.id.tolist(), rotation=-90)
    ax.tick_params('x', labelsize=6)
    ax.tick_params('y', labelsize=12)

    ax.set_xlim([lim_x_min, lim_x_max])
    ax.set_ylim([lim_y_min, lim_y_max])

    dot_radius_x = (dot_radius_in / fig_size_y_in *
                    ax_size_y_ratio) * ax_width

    for x, y in zip(df.ORD, df.value):
        ax.add_patch(
            patches.Ellipse(
                (x, y),
                dot_radius_x,
                dot_radius_x / ((fig_size_y_in * ax_size_y_ratio / ax_height) /
                                (fig_size_x_in * ax_size_x_ratio / ax_width)),
                clip_on=False,
                zorder=200,
                color='tab:orange'
            )
        )
        ax.add_patch(
            patches.Ellipse(
                (x, y),
                dot_radius_x,
                dot_radius_x / ((fig_size_y_in * ax_size_y_ratio / ax_height) /
                                (fig_size_x_in * ax_size_x_ratio / ax_width)),
                clip_on=False,
                fill=False,
                zorder=200,
                linewidth=2,
            )
        )


def add_group_sep(df, ax, group_id, level, para, mode='line'):

    ax_height = para['lim_y_max'] - para['lim_y_min']
    sep, group = get_sep(df, group_id)
    if mode == 'line':
        for s0, s1, gr in zip(sep[1:], sep[:-1], group):
            codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
            y0 = (-igroup_indi_offset_y_in - igroup_indi_height_y_in *
                  level - igroup_indi_sep_y_in / 2)
            y1 = (-igroup_indi_offset_y_in - igroup_indi_height_y_in *
                  (level + 1) + igroup_indi_sep_y_in / 2)
            vertices = [(s0, y0 / (fig_size_y_in * ax_size_y_ratio / ax_height)),
                        (s0, y1 / (fig_size_y_in * ax_size_y_ratio / ax_height)),
                        (s1, y0 / (fig_size_y_in * ax_size_y_ratio / ax_height)),
                        (s1, y1 / (fig_size_y_in * ax_size_y_ratio / ax_height))]
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
            ax.text(s0 / 2 + s1 / 2, (-igroup_indi_offset_y_in - igroup_indi_height_y_in * (level + 0.5) - igroup_indi_sep_y_in / 2) /
                    (fig_size_y_in * ax_size_y_ratio / ax_height), gr, ha='center', va='center')
    elif mode == 'box':
        if 'color_table' not in para.keys():
            para['color_table'] = dict()
        color_idx = 0
        for s0, s1, gr in zip(sep[1:], sep[:-1], group):
            if gr not in para['color_table'].keys():
                while 'C' + str(color_idx) in para['color_table'].values():
                    color_idx += 1
                para['color_table'][gr] = 'C' + str(color_idx)
            ax.add_patch(
                patches.Rectangle(
                    (s0, (-igroup_indi_offset_y_in - igroup_indi_height_y_in * (level + 1)) /
                        (fig_size_y_in * ax_size_y_ratio / ax_height)),
                    s1 - s0,
                    igroup_indi_height_y_in /
                    (fig_size_y_in * ax_size_y_ratio / ax_height),
                    color=para['color_table'][gr],
                    clip_on=False,
                    zorder=200,
                    linewidth=0.75,
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (s0, (-igroup_indi_offset_y_in - igroup_indi_height_y_in * (level + 1)) /
                     (fig_size_y_in * ax_size_y_ratio / ax_height)),
                    s1 - s0,
                    igroup_indi_height_y_in /
                    (fig_size_y_in * ax_size_y_ratio / ax_height),
                    fill=False,
                    clip_on=False,
                    zorder=200,
                    linewidth=0.75,
                )
            )
            ax.text(s0 / 2 + s1 / 2, (-igroup_indi_offset_y_in - igroup_indi_height_y_in * (level + 0.5) - igroup_indi_sep_y_in / 2) /
                    (fig_size_y_in * ax_size_y_ratio / ax_height), gr, ha='center', va='center', zorder=201)

    return


def add_spec_line(df, ax, spec, label, para, col='r'):
    ax.plot([para['lim_x_min'], para['lim_x_max']],
            [spec, spec], '--', linewidth=1.0, zorder=1, color=col)
    ax.text(0, spec, label, ha='left', va='bottom', color=col)
    return


def get_sep(df, group_id):

    gn = df[group_id].tolist()
    current = gn[0]
    sep = [0]
    group = []
    for idx, n in enumerate(gn[1:]):
        if current != n:
            sep.append(idx + 1)
            group.append(current)
            current = n

    sep.append(idx + 2)
    group.append(current)
    return sep, group


def readdata(para):

    fn = para['input'][0]
    try:
        out = pd.read_excel(fn, sheet_name='Sheet1')
    except:
        out = pd.read_csv(fn)

    return out


def groupdata(df, para):

    if (para['group']) != None:
        df.sort_values(by=para['group'], inplace=True)
    df['ORD'] = np.arange(0.5, len(df.id.tolist()) + 0.5)

    return


def main(para):

    # read data to dataframe
    df = readdata(para)

    # sort dataframe by group
    groupdata(df, para)

    # build dict for plot parameters
    dic_plot_para = dict()

    dic_plot_para['lim_x_min'] = 0
    dic_plot_para['lim_x_max'] = len(df.id.tolist())
    dic_plot_para['lim_y_min'] = 0
    if para['ymax'] != None:
        dic_plot_para['lim_y_max'] = para['ymax'][0]
    else:
        dic_plot_para['lim_y_max'] = max(df.value)

    # build figure
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=[fig_size_x_in, fig_size_y_in], dpi=fig_dpi)
    ax = fig.add_axes(
        [ax_offset_x_ratio, ax_offset_y_ratio, ax_size_x_ratio, ax_size_y_ratio])

    # plot base plot
    tcbase(df, ax, dic_plot_para)

    # add spec line
    for spec_name, spec_value in zip(para['spec'][::2], para['spec'][1::2]):
        add_spec_line(df, ax, int(spec_value), spec_name, dic_plot_para)

    # add grouping indicators
    if (para['group']) != None:
        for level, group_id in enumerate(para['group']):
            add_group_sep(df, ax, group_id, len(para['group']) - level - 1,
                          mode=para['groupstyle'][0], para=dic_plot_para)

    # save figure to file
    plt.savefig(para['output'][0])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Plot pivot charts from data')

    parser.add_argument('input', metavar='INPUT',
                        nargs=1, help="input excel or csv file")
    parser.add_argument('-c', metavar='CONDITION',
                        nargs=1, help="condition excel or csv file")
    parser.add_argument('-o', '--output', metavar='OUTPUT',
                        nargs=1, help="output file path")
    parser.add_argument('-x', metavar='X',
                        nargs=1, help="data x")
    parser.add_argument('-y', metavar='Y',
                        nargs=1, help="data y")
    parser.add_argument('--ymax', metavar='YMAX',
                        nargs=1, type=int, help="data y upper limit")
    parser.add_argument('-g', '--group', metavar='GROUPS',
                        nargs='+', help="grouping condition")
    parser.add_argument('--groupstyle', metavar='STYLE', default=['box'], type=str,
                        nargs=1, help="grouping style (line or box)")
    parser.add_argument('--spec', metavar='SPEC',
                        nargs='+', help="add spec indication")

    args = parser.parse_args()
    main(vars(args))
