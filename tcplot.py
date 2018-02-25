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
t_DEFAULT = 'time'
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


def plot_data(df, ax, para):

    ax_height = para['ymax'] - para['ymin']
    ax_width = para['xmax'] - para['xmin']

    ax.plot(df.ORD, df['scaled_value'], 'k', linewidth=2.0, zorder=1)
    # ax.bar(df.ORD, df['scaled_value'], zorder=1)
    plt.xticks(df.ORD, df[para['x']].tolist(), rotation=90)
    ax.tick_params('x', labelsize=6)
    ax.tick_params('y', labelsize=6)

    ax.set_xlim([para['xmin'], para['xmax']])
    ax.set_ylim([para['ymin'], para['ymax']])

    dot_radius_x = (dot_radius_in / fig_size_y_in *
                    ax_size_y_ratio) * ax_width

    for x, y in zip(df.ORD, df['scaled_value']):
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
    sep, group = cal_sep(df, group_id)
    if mode == 'line':
        y0 = (-igroup_indi_offset_y_in - igroup_indi_height_y_in *
              level - igroup_indi_sep_y_in / 2)
        y1 = (-igroup_indi_offset_y_in - igroup_indi_height_y_in *
              (level + 1) + igroup_indi_sep_y_in / 2)
        for s0, s1, gr in zip(sep + [para['xmax']], [0] + sep, group):
            ax.text(s0 / 2 + s1 / 2, (-igroup_indi_offset_y_in - igroup_indi_height_y_in * (level + 0.5) - igroup_indi_sep_y_in / 2) /
                    (fig_size_y_in * ax_size_y_ratio / ax_height), gr, ha='center', va='center')
        codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
        vertices = [(0, y0 / (fig_size_y_in * ax_size_y_ratio / ax_height)),
                    (0, y1 / (fig_size_y_in * ax_size_y_ratio / ax_height)),
                    (para['xmax'], y0 / (fig_size_y_in *
                                         ax_size_y_ratio / ax_height)),
                    (para['xmax'], y1 / (fig_size_y_in * ax_size_y_ratio / ax_height))]
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
        for s in sep:
            ax.plot([s, s], [para['ymin'], para['ymax']],
                    '--', linewidth=1.0, zorder=1, color='tab:gray')
            codes = [Path.MOVETO, Path.LINETO]
            vertices = [(s, y0 / (fig_size_y_in * ax_size_y_ratio / ax_height)),
                        (s, y1 / (fig_size_y_in * ax_size_y_ratio / ax_height))]
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
        if 'color_table' not in para.keys():
            para['color_table'] = dict()
        color_idx = 0
        for s in sep:
            ax.plot([s, s], [para['ymin'], para['ymax']],
                    '--', linewidth=1.0, zorder=1, color='tab:gray')
        for s0, s1, gr in zip(sep + [para['xmax']], [0] + sep, group):
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
                    fc=para['color_table'][gr],
                    ec='k',
                    clip_on=False,
                    zorder=200,
                )
            )
            ax.text(s0 / 2 + s1 / 2, (-igroup_indi_offset_y_in - igroup_indi_height_y_in * (level + 0.5) - igroup_indi_sep_y_in / 2) /
                    (fig_size_y_in * ax_size_y_ratio / ax_height), gr, ha='center', va='center', zorder=201)

    return


def add_spec_line(df, ax, spec, label, para, col='r'):

    ax.plot([para['xmin'], para['xmax']],
            [spec, spec], '--', lw=1.0, zorder=1, color=col)
    ax.text(0, spec, '{:}: {:}'.format(label, spec),
            ha='left', va='bottom', color=col)

    return


def cal_sep(df, group_id):

    gn = df[group_id].tolist()

    current = gn[0]
    sep = []
    group = []

    for idx, n in enumerate(gn[1:]):
        if current != n:
            sep.append(idx + 1)
            group.append(current)
            current = n
    group.append(current)

    return sep, group


def read_data(para):

    out = pd.read_excel(para['input'], sheet_name=0)

    for fn_condi in para['condition']:
        out2 = pd.read_excel(fn_condi, sheet_name=0)
        out = pd.merge(out2, out, how='outer', on=para['x'])

    out.fillna('NA', inplace=True)
    out['scaled_value'] = out[para['y']] * para['yscale']

    return out


def group_data(df, para):

    if (para['group']) != None:
        allgroup = []
        for x in para['group']:
            if x in df.columns:
                allgroup.append(x)
        if len(allgroup) == 0:
            if para['t'] in df.columns:
                df.sort_values(by=para['t'], inplace=True)
            para['group'] = None
        else:
            if para['t'] in df.columns:
                df.sort_values(by=allgroup + [para['t']], inplace=True)
            else:
                df.sort_values(by=allgroup, inplace=True)
            para['group'] = allgroup

    df['ORD'] = np.arange(0.5, len(df[para['x']].tolist()) + 0.5)

    return


def adjust_yticks(ax, para):

    tmp = para['ymax']
    ex = 0

    while tmp > 10:
        tmp /= 10
        ex += 1

    if ex == 0:
        sep = 1
    else:
        if tmp > 8:
            sep = 2
        elif tmp > 6:
            sep = 1.5
        elif tmp > 3:
            sep = 1
        elif tmp >= 1.5:
            sep = 0.5
        elif tmp <= 1.5:
            sep = 0.2

    n = int(tmp // sep)

    if ex > 2:
        outformat = '{:.1E}'
    else:
        outformat = '{:g}'
    outyticks = [(sep * (m + 1) * 10**ex) for m in range(n)]
    outytick_labels = [outformat.format(m) for m in outyticks]
    plt.yticks(outyticks, outytick_labels)

    return


def main(para):

    # read data to dataframe
    df = read_data(para)

    # set plot parameters
    para['xmin'] = 0
    para['xmax'] = len(df[para['x']].tolist())
    para['ymin'] = 0
    if para['ymax'] != None:
        para['ymax'] = para['ymax']
    else:
        para['ymax'] = max(df['scaled_value'])

    # sort dataframe by group
    group_data(df, para)

    # build figure
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=[fig_size_x_in, fig_size_y_in], dpi=fig_dpi)
    ax = fig.add_axes(
        [ax_offset_x_ratio, ax_offset_y_ratio, ax_size_x_ratio, ax_size_y_ratio])

    # plot base plot
    plot_data(df, ax, para)

    # add spec line
    if para['spec'] is not None:
        for spec_name, spec_value in zip(para['spec'][::2], para['spec'][1::2]):
            add_spec_line(df, ax, int(spec_value), spec_name, para)

    # add grouping indicators
    if (para['group']) is not None:
        for level, group_id in enumerate(para['group']):
            add_group_sep(df, ax, group_id, len(para['group']) - level - 1,
                          mode=para['groupstyle'], para=para)

    adjust_yticks(ax, para)

    # save figure to file
    plt.savefig(para['output'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Plot pivot charts from data')

    parser.add_argument('-i', '--input', metavar='INPUT', default=[input_DEFAULT],
                        nargs=1, help="input excel or csv file")
    parser.add_argument('-c', '--condition', metavar='CONDITION', default=[],
                        nargs='+', help="condition excel or csv file")
    parser.add_argument('-o', '--output', metavar='OUTPUT',
                        nargs=1, help="output file path")
    parser.add_argument('-x', metavar='X', default=[x_DEFAULT],
                        nargs=1, help="data x")
    parser.add_argument('-y', metavar='Y', default=[y_DEFAULT],
                        nargs=1, help="data y")
    parser.add_argument('--yscale', metavar='YSCALE', default=[1.0],
                        nargs=1, type=float, help="scaling factor for y")
    parser.add_argument('--ymax', metavar='YMAX',
                        nargs=1, type=float, help="data y upper limit")
    parser.add_argument('-t', metavar='TIME', default=[t_DEFAULT],
                        nargs=1, help="data time")
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
    args['t'] = args['t'][0]
    args['yscale'] = args['yscale'][0]
    args['input'] = args['input'][0]
    if args['output'] is not None:
        args['output'] = args['output'][0]
    args['groupstyle'] = args['groupstyle'][0]

    if args['ymax'] is not None:
        args['ymax'] = args['ymax'][0]

    if args['gui']:

        # main containers
        root = Tk()
        root.resizable(width=False, height=False)

        frame = Frame(root, padx=4, pady=4)
        frame.grid(row=0, column=0)
        row0 = LabelFrame(frame, text="input", padx=4, pady=4)
        row0.grid(row=0, column=0, sticky=W)
        row1 = LabelFrame(frame, text="data", padx=4, pady=4)
        row1.grid(row=1, column=0, sticky=W)
        row2 = LabelFrame(frame, text="time", padx=4, pady=4)
        row2.grid(row=2, column=0, sticky=W)
        row3 = LabelFrame(frame, text="grouping", padx=4, pady=4)
        row3.grid(row=3, column=0, sticky=W)
        row4 = LabelFrame(frame, text="spec", padx=4, pady=4)
        row4.grid(row=4, column=0, sticky=W)

        to_save = []

        # helper functions
        def get_wd():
            tmp = fd.askdirectory()
            if tmp != '':
                input_dirname.set(tmp)
            return

        def run():
            if input_dirname.get() != '':
                args['input'] = op.join(input_dirname.get(), input_DEFAULT)
                if args['output'] is None:
                    args['output'] = op.join(
                        input_dirname.get(), output_DEFAULT)
                if data_x.get() != '':
                    args['x'] = data_x.get()
                if data_y.get() != '':
                    args['y'] = data_y.get()
                if data_yscale.get() != '':
                    try:
                        tmp = float(data_yscale.get())
                        args['yscale'] = tmp
                    except:
                        pass
                if data_ymax.get() != '':
                    try:
                        tmp = float(data_ymax.get())
                        args['ymax'] = tmp
                    except:
                        pass
                if time_t.get() != '':
                    args['t'] = time_t.get()
                if group_enable.get():
                    args['groupstyle'] = {0: "box",
                                          1: "line"}[group_style.get()]
                    args['group'] = [group_type0.get(), group_type1.get(),
                                     group_type2.get()]
                if spec_enable.get():
                    args['spec'] = []
                    if (spec_name0.get() != '') and (spec_value0.get() != ''):
                        args['spec'].append(spec_name0.get())
                        args['spec'].append(spec_value0.get())
                    if (spec_name1.get() != '') and (spec_value1.get() != ''):
                        args['spec'].append(spec_name1.get())
                        args['spec'].append(spec_value1.get())

                main(args)
            return

        def load_config():
            if input_dirname.get() == '':
                return
            try:
                tmp = pd.read_excel(
                    op.join(input_dirname.get(), 'config.xlsx'), sheet_name='Sheet1')
                tmp.fillna('', inplace=True)
            except:
                return
            for vpair in tmp[['vname', 'vvalue', 'vtype']].values.tolist():
                vname = str(vpair[0])
                vvalue = str(vpair[1])
                vtype = str(vpair[2])
                try:
                    if vtype == 'String':
                        expr = vname + '.set(\'' + vvalue + '\')'
                    else:
                        expr = vname + '.set(' + vvalue + ')'
                    exec(expr)
                except:
                    print('fail to exec \'{:}\''.format(expr))
                    pass
            return

        def save_config():
            if input_dirname.get() == '':
                return
            vname = []
            vvalue = []
            vtype = []
            for vpair in to_save:  # name, type
                vname.append(vpair[0])
                vtype.append(vpair[1])
                vvalue.append(str(eval(vpair[0] + '.get()')))
            tmpdf = pd.DataFrame(
                {'vname': vname, 'vvalue': vvalue, 'vtype': vtype})
            tmpdf.to_excel(
                op.join(input_dirname.get(), 'config.xlsx'), sheet_name='Sheet1', index=False)
            return

        # row 0
        input_dirname = StringVar()

        input_b0 = Button(row0, width=4, text="run", command=run)
        input_b1 = Button(row0, width=6, text="folder", command=get_wd)
        input_b2 = Button(row0, width=10, text="load config",
                          command=load_config)
        input_b3 = Button(row0, width=10, text="save config",
                          command=save_config)
        input_et0 = Entry(row0, width=58, textvariable=input_dirname)

        input_b0.grid(row=0, column=0)
        input_et0.grid(row=0, column=1)
        input_b1.grid(row=0, column=2)
        input_b2.grid(row=0, column=3)
        input_b3.grid(row=0, column=4)

        # row 1
        data_x = StringVar()
        data_y = StringVar()
        data_yscale = StringVar()
        data_ymax = StringVar()

        to_save.append(['data_x', 'String'])
        to_save.append(['data_y', 'String'])
        to_save.append(['data_yscale', 'String'])
        to_save.append(['data_ymax', 'String'])

        data_et0 = Entry(row1, width=8, textvariable=data_x)
        data_et1 = Entry(row1, width=8, textvariable=data_y)
        data_et2 = Entry(row1, width=8, textvariable=data_yscale)
        data_et3 = Entry(row1, width=8, textvariable=data_ymax)

        Label(row1, text='x').grid(row=0, column=0)
        data_et0.grid(row=0, column=1)
        Label(row1, text='y').grid(row=0, column=2)
        data_et1.grid(row=0, column=3)
        Label(row1, text='yscale').grid(row=0, column=4)
        data_et2.grid(row=0, column=5)
        Label(row1, text='ymax').grid(row=0, column=6)
        data_et3.grid(row=0, column=7)

        # row 2
        time_t = StringVar()
        to_save.append(['time_t', 'String'])

        time_et0 = Entry(row2, width=8, textvariable=time_t)
        Label(row2, text='t').grid(row=0, column=0)
        time_et0.grid(row=0, column=1)

        # row 3
        group_enable = BooleanVar()
        group_style = IntVar()
        group_type0 = StringVar()
        group_type1 = StringVar()
        group_type2 = StringVar()

        to_save.append(['group_enable', 'Boolean'])
        to_save.append(['group_style', 'Int'])
        to_save.append(['group_type0', 'String'])
        to_save.append(['group_type1', 'String'])
        to_save.append(['group_type2', 'String'])

        group_cb = Checkbutton(row3, text="enable", variable=group_enable)
        group_rb0 = Radiobutton(
            row3, text="box", variable=group_style, value=0)
        group_rb1 = Radiobutton(
            row3, text="line", variable=group_style, value=1)
        group_et0 = Entry(row3, width=8, textvariable=group_type0)
        group_et1 = Entry(row3, width=8, textvariable=group_type1)
        group_et2 = Entry(row3, width=8, textvariable=group_type2)

        group_cb.grid(row=0, column=0)
        group_rb0.grid(row=0, column=1)
        group_rb1.grid(row=0, column=2)
        Label(row3, text='group 1').grid(row=0, column=3)
        group_et0.grid(row=0, column=4)
        Label(row3, text='group 2').grid(row=0, column=5)
        group_et1.grid(row=0, column=6)
        Label(row3, text='group 3').grid(row=0, column=7)
        group_et2.grid(row=0, column=8)

        # row 4
        spec_enable = BooleanVar()
        spec_name0 = StringVar()
        spec_value0 = StringVar()
        spec_name1 = StringVar()
        spec_value1 = StringVar()

        to_save.append(['spec_enable', 'Boolean'])
        to_save.append(['spec_name0', 'String'])
        to_save.append(['spec_value0', 'String'])
        to_save.append(['spec_name1', 'String'])
        to_save.append(['spec_value1', 'String'])

        spec_cb = Checkbutton(row4, text="enable", variable=spec_enable)
        spec_et0 = Entry(row4, width=8, textvariable=spec_name0)
        spec_et1 = Entry(row4, width=16, textvariable=spec_value0)
        spec_et2 = Entry(row4, width=8, textvariable=spec_name1)
        spec_et3 = Entry(row4, width=16, textvariable=spec_value1)

        spec_cb.grid(row=0, column=0)
        Label(row4, text='spec 1').grid(row=0, column=1)
        spec_et0.grid(row=0, column=2)
        spec_et1.grid(row=0, column=3)
        Label(row4, text='spec 2').grid(row=0, column=4)
        spec_et2.grid(row=0, column=5)
        spec_et3.grid(row=0, column=6)

        # start gui
        mainloop()

    else:

        main(args)
