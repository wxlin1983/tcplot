import pandas as pd
import numpy as np
import os.path as op
from tkinter import *
from tkinter import filedialog as fd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

import argparse

input_DEFAULT = 'data.xlsx'
output_DEFAULT = 'output.png'
x_DEFAULT = 'id'
y_DEFAULT = 'value'
yscale_DEFAULT = 1

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

    ax_height = para['ymax'] - para['ymin']
    ax_width = para['xmax'] - para['xmin']

    ax.plot(df.ORD, df.value, 'k', linewidth=2.0, zorder=1)
    plt.xticks(df.ORD, df[para['x']].tolist(), rotation=-90)
    ax.tick_params('x', labelsize=6)
    ax.tick_params('y', labelsize=12)

    ax.set_xlim([para['xmin'], para['xmax']])
    ax.set_ylim([para['ymin'], para['ymax']])

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

    ax_height = para['ymax'] - para['ymin']
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
    ax.plot([para['xmin'], para['xmax']],
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

    fn = para['input']
    try:
        out = pd.read_excel(fn, sheet_name='Sheet1')
    except:
        out = pd.read_csv(fn)

    for fn_condi in para['condition']:
        out2 = pd.read_excel(fn_condi, sheet_name='Sheet1')
        out = pd.merge(out2, out, how='outer', on=para['x'])

    out.fillna('NA', inplace=True)

    return out


def groupdata(df, para):

    if (para['group']) != None:
        allgroup = []
        for x in para['group']:
            if x in df.columns:
                allgroup.append(x)
        if len(allgroup) == 0:
            para['group'] = None
        else:
            para['group'] = allgroup
            df.sort_values(by=para['group'], inplace=True)
    df['ORD'] = np.arange(0.5, len(df[para['x']].tolist()) + 0.5)

    return


def main(para):

    # read data to dataframe
    df = readdata(para)

    # sort dataframe by group
    groupdata(df, para)

    # build dict for plot parameters
    dic_plot_para = dict()

    para['xmin'] = 0
    para['xmax'] = len(df[para['x']].tolist())
    para['ymin'] = 0
    if para['ymax'] != None:
        para['ymax'] = para['ymax']
    else:
        para['ymax'] = max(df.value)

    # build figure
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=[fig_size_x_in, fig_size_y_in], dpi=fig_dpi)
    ax = fig.add_axes(
        [ax_offset_x_ratio, ax_offset_y_ratio, ax_size_x_ratio, ax_size_y_ratio])

    # plot base plot
    tcbase(df, ax, para)

    # add spec line
    if para['spec'] is not None:
        for spec_name, spec_value in zip(para['spec'][::2], para['spec'][1::2]):
            add_spec_line(df, ax, int(spec_value), spec_name, dic_plot_para)

    # add grouping indicators
    if (para['group']) is not None:
        for level, group_id in enumerate(para['group']):
            add_group_sep(df, ax, group_id, len(para['group']) - level - 1,
                          mode=para['groupstyle'][0], para=para)

    # save figure to file
    plt.savefig(para['output'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Plot pivot charts from data')

    parser.add_argument('-i', '--input', metavar='INPUT', default=[input_DEFAULT],
                        nargs=1, help="input excel or csv file")
    parser.add_argument('-c', '--condition', metavar='CONDITION', default=[],
                        nargs='+', help="condition excel or csv file")
    parser.add_argument('-o', '--output', metavar='OUTPUT', default=[output_DEFAULT],
                        nargs=1, help="output file path")
    parser.add_argument('-x', metavar='X', default=[x_DEFAULT],
                        nargs=1, help="data x")
    parser.add_argument('-y', metavar='Y', default=[y_DEFAULT],
                        nargs=1, help="data y")
    parser.add_argument('--yscale', metavar='YSCALE',
                        nargs=1, type=float, help="scaling factor for y")
    parser.add_argument('--ymax', metavar='YMAX',
                        nargs=1, type=int, help="data y upper limit")
    parser.add_argument('-g', '--group', metavar='GROUPS',
                        nargs='+', help="grouping condition")
    parser.add_argument('--groupstyle', metavar='STYLE', default=['box'], type=str,
                        nargs=1, help="grouping style (line or box)")
    parser.add_argument('--spec', metavar='SPEC',
                        nargs='+', help="add spec indication")
    parser.add_argument('--gui', action='store_true')

    args = vars(parser.parse_args())

    args['x'] = args['x'][0]
    args['y'] = args['y'][0]
    args['input'] = args['input'][0]
    args['output'] = args['output'][0]

    if args['ymax'] is not None:
        args['ymax'] = args['ymax'][0]

    if args['yscale'] is not None:
        args['yscale'] = args['yscale'][0]

    print(args)

    if args['gui']:

        root = Tk()
        root.resizable(width=False, height=False)

        frame = Frame(root, padx=4, pady=4)
        frame.grid(row=0, column=0)
        row0 = LabelFrame(frame, text="input", padx=4, pady=4)
        row0.grid(row=0, column=0, sticky=W)
        row1 = LabelFrame(frame, text="data", padx=4, pady=4)
        row1.grid(row=1, column=0, sticky=W)
        row2 = LabelFrame(frame, text="grouping", padx=4, pady=4)
        row2.grid(row=2, column=0, sticky=W)
        row3 = LabelFrame(frame, text="spec", padx=4, pady=4)
        row3.grid(row=3, column=0, sticky=W)

        # row 0
        def get_wd():
            wdn.set(fd.askdirectory())
            return

        def run():
            if wdn.get() != '':
                args['input'] = op.join(wdn.get(), input_DEFAULT)
                if args['output'] is None:
                    args['output'] = op.join(wdn.get(), output_DEFAULT)
                if toGroup.get():
                    args['groupstyle'] = {0: ["box"],
                                          1: ["line"]}[grouptype.get()]
                    args['group'] = [group_type_0.get(), group_type_1.get(),
                                     group_type_2.get()]
                if toSPEC.get():
                    args['spec'] = []
                    if (spec_name_0.get() != '') and (spec_value_0.get() != ''):
                        args['spec'].append(spec_name_0.get())
                        args['spec'].append(spec_value_0.get())
                    if (spec_name_1.get() != '') and (spec_value_1.get() != ''):
                        args['spec'].append(spec_name_1.get())
                        args['spec'].append(spec_value_1.get())

                main(args)
            return

        wdn = StringVar()
        wd_button = Button(
            row0, width=8, text="folder", command=get_wd)
        wd_button.grid(row=0, column=1)
        wd_entry = Entry(row0, width=58, textvariable=wdn)
        wd_entry.grid(row=0, column=2)

        run_button = Button(
            row0, width=8, text="run", command=run)
        run_button.grid(row=0, column=0)

        # row 1
        wg_0_x = Entry(row1, width=8)
        wg_0_y = Entry(row1, width=8)
        wg_0_ymax = Entry(row1, width=8)

        Label(row1, text='x').grid(row=0, column=0)
        wg_0_x.grid(row=0, column=1)
        Label(row1, text='y').grid(row=0, column=2)
        wg_0_y.grid(row=0, column=3)
        Label(row1, text='ymax').grid(row=0, column=4)
        wg_0_ymax.grid(row=0, column=5)

        # row 2
        toGroup = BooleanVar()
        grouptype = IntVar()

        wg_1_c1 = Checkbutton(row2, text="enable", variable=toGroup)
        wg_1_r1 = Radiobutton(row2, text="box", variable=grouptype, value=0)
        wg_1_r2 = Radiobutton(row2, text="line", variable=grouptype, value=1)

        wg_1_c1.grid(row=0, column=0)
        wg_1_r1.grid(row=0, column=1)
        wg_1_r2.grid(row=0, column=2)

        group_type_0 = StringVar()
        group_type_1 = StringVar()
        group_type_2 = StringVar()

        wg_1_group_type_0 = Entry(row2, width=8, textvariable=group_type_0)
        wg_1_group_type_1 = Entry(row2, width=8, textvariable=group_type_1)
        wg_1_group_type_2 = Entry(row2, width=8, textvariable=group_type_2)

        Label(row2, text='group 1').grid(row=0, column=3)
        wg_1_group_type_0.grid(row=0, column=4)
        Label(row2, text='group 2').grid(row=0, column=5)
        wg_1_group_type_1.grid(row=0, column=6)
        Label(row2, text='group 3').grid(row=0, column=7)
        wg_1_group_type_2.grid(row=0, column=8)

        # row 3
        toSPEC = BooleanVar()

        wg_2_c1 = Checkbutton(row3, text="enable", variable=toSPEC)

        wg_2_c1.grid(row=0, column=0)

        spec_name_0 = StringVar()
        spec_value_0 = StringVar()
        spec_name_1 = StringVar()
        spec_value_1 = StringVar()

        wg_2_spec_name_0 = Entry(row3, width=8, textvariable=spec_name_0)
        wg_2_spec_value_0 = Entry(row3, width=16, textvariable=spec_value_0)
        wg_2_spec_name_1 = Entry(row3, width=8, textvariable=spec_name_1)
        wg_2_spec_value_1 = Entry(row3, width=16, textvariable=spec_value_1)

        Label(row3, text='spec 1').grid(row=0, column=1)
        wg_2_spec_name_0.grid(row=0, column=2)
        wg_2_spec_value_0.grid(row=0, column=3)
        Label(row3, text='spec 2').grid(row=0, column=4)
        wg_2_spec_name_1.grid(row=0, column=5)
        wg_2_spec_value_1.grid(row=0, column=6)

        mainloop()

    else:

        main(args)
