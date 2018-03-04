import pandas as pd
import numpy as np
import os.path as op
import logging

from tkinter import *
from tkinter import filedialog as fd

import matplotlib.pyplot as plt
import matplotlib.patches as pch
from matplotlib.path import Path as pth

import argparse


input_DEFAULT = 'data.xlsx'
output_DEFAULT = 'output.png'
profile_DEFAULT = 'default.xlsx'
fc_color_DEFAULT = 'tab:orange'

t_DEFAULT = 'time'
x_DEFAULT = 'id'
y_DEFAULT = 'value'
yscale_DEFAULT = 1

fig_size_x_in = 10
fig_size_y_in = 2.5

ax_size_x_in = 9.25
ax_size_y_in = 1.25
ax_padding_x_left_in = 0.5
ax_padding_y_bottom_in = 1.125

dot_radius_in = 0.1
dot_radius_min_in = 0.05

ax_size_x_ratio = ax_size_x_in / fig_size_x_in
ax_size_y_ratio = ax_size_y_in / fig_size_y_in

igroup_indi_offset_y_in = 0.6
igroup_indi_height_y_in = 0.2
igroup_indi_sep_y_in = 0.02

fig_dpi = 300


def plot_data(df, ax, para):

    ax_height = para['ymax'] - para['ymin']
    ax_width = para['xmax'] - para['xmin']

    if para['chartstyle'] == 'chart':
        ax.plot(df.ORD, df['scaled_value'], 'k', linewidth=2.0, zorder=1)
    elif para['chartstyle'] == 'bar':
        ax.bar(df.ORD, df['scaled_value'], zorder=1)

    plt.xticks(df.ORD, df[para['x']].tolist(), rotation=90)
    ax.tick_params('x', labelsize=6)
    ax.tick_params('y', labelsize=8)

    ax.set_xlim([para['xmin'], para['xmax']])
    ax.set_ylim([para['ymin'], para['ymax']])

    # adjust dot size
    my_dot_radius_in = dot_radius_in
    tmpdot = 1.5 * ax_size_x_in / len(df.ORD.tolist()) / 2

    if my_dot_radius_in > tmpdot:
        my_dot_radius_in = tmpdot
    if my_dot_radius_in < dot_radius_min_in:
        my_dot_radius_in = dot_radius_min_in

    # build dot color table
    if para['chartstyle'] == 'chart':
        color_idx = 0
        if para['color'] in df.columns:
            for gr in df[para['color']].unique():
                if gr not in para['color_table'].keys():
                    while 'C' + str(color_idx) in para['color_table'].values():
                        color_idx += 1
                    para['color_table'][gr] = 'C' + str(color_idx)

    # add dot if chartstyle is chart
    if para['chartstyle'] == 'chart':
        dot_radius_x = (my_dot_radius_in / fig_size_y_in *
                        ax_size_y_ratio) * ax_width

        if para['color'] in df.columns:
            color_columns = para['color']
        else:
            color_columns = 'ORD'

        for x, y, z in zip(df.ORD, df['scaled_value'], df[color_columns]):

            if color_columns == 'ORD':
                fc_color = fc_color_DEFAULT
            else:
                fc_color = para['color_table'][z]

            ax.add_patch(
                pch.Ellipse(
                    (x, y),
                    dot_radius_x,
                    dot_radius_x / ((fig_size_y_in * ax_size_y_ratio / ax_height) /
                                    (fig_size_x_in * ax_size_x_ratio / ax_width)),
                    clip_on=False,
                    zorder=200,
                    fc=fc_color,
                    ec='k',
                    linewidth=2
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
        if not para['grouptextblank']:
            for s0, s1, gr in zip(sep + [para['xmax']], [0] + sep, group):
                ax.text(s0 / 2 + s1 / 2, (-igroup_indi_offset_y_in - igroup_indi_height_y_in * (level + 0.5) - igroup_indi_sep_y_in / 2) /
                        (fig_size_y_in * ax_size_y_ratio / ax_height), gr, ha='center', va='center')
        codes = [pth.MOVETO, pth.LINETO, pth.MOVETO, pth.LINETO]
        vertices = [(0, y0 / (fig_size_y_in * ax_size_y_ratio / ax_height)),
                    (0, y1 / (fig_size_y_in * ax_size_y_ratio / ax_height)),
                    (para['xmax'], y0 / (fig_size_y_in *
                                         ax_size_y_ratio / ax_height)),
                    (para['xmax'], y1 / (fig_size_y_in * ax_size_y_ratio / ax_height))]
        vertices = np.array(vertices, float)
        mypath = pth(vertices, codes)
        ax.add_patch(
            pch.PathPatch(
                mypath,
                clip_on=False,
                zorder=200,
                linewidth=0.75,
            )
        )
        for s in sep:
            ax.plot([s, s], [para['ymin'], para['ymax']],
                    '--', linewidth=1.0, zorder=1, color='tab:gray')
            codes = [pth.MOVETO, pth.LINETO]
            vertices = [(s, y0 / (fig_size_y_in * ax_size_y_ratio / ax_height)),
                        (s, y1 / (fig_size_y_in * ax_size_y_ratio / ax_height))]
            vertices = np.array(vertices, float)
            mypath = pth(vertices, codes)
            ax.add_patch(
                pch.PathPatch(
                    mypath,
                    clip_on=False,
                    zorder=200,
                    linewidth=0.75,
                )
            )
    elif mode == 'box':
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
                pch.Rectangle(
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
            if not para['grouptextblank']:
                ax.text(s0 / 2 + s1 / 2, (-igroup_indi_offset_y_in - igroup_indi_height_y_in * (level + 0.5) - igroup_indi_sep_y_in / 2) /
                        (fig_size_y_in * ax_size_y_ratio / ax_height), gr, ha='center', va='center', zorder=201)

    return


def add_spec_line(df, ax, spec, label, para, col='r'):

    logger.info('adding spec line: {:}: {:}'.format(label, spec))
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
    logger.info('loaded data from \"{:}\".'.format(para['input']))

    for fn_condi in para['condition']:
        try:
            out2 = pd.read_excel(fn_condi, sheet_name=0)
            out = pd.merge(out2, out, how='outer', on=para['x'])
            logger.info('loaded conditions from \"{:}\".'.format(fn_condi))
        except:
            logger.error(
                'cannot load conditions from \"{:}\".'.format(fn_condi))
            para['condition'].remove(fn_condi)

    out.fillna('NA', inplace=True)
    out['scaled_value'] = out[para['y']] * para['yscale']

    return out


def group_data(df, para):

    logger.info('grouping data.')

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

    logger.info('adjusting yick numbers.')
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
        elif tmp > 1.5:
            sep = 0.5
        else:
            sep = 0.2

    n = int(tmp // sep)

    outyticks = [(sep * (m + 1) * 10**ex) for m in range(n)]
    if ex > 2:
        outytick_labels = ['{:.1E}'.format(m) for m in outyticks]
        outytick_labels = [m[:-3] + m[-1] for m in outytick_labels]
    else:
        outytick_labels = ['{:g}'.format(m) for m in outyticks]
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
        [ax_padding_x_left_in / fig_size_x_in, ax_padding_y_bottom_in / fig_size_y_in, ax_size_x_ratio, ax_size_y_ratio])

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

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('tcplot.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(levelname)s][%(asctime)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser(
        description='Plot pivot charts from data')

    parser.add_argument('-i', '--input', metavar='INPUT', default=[input_DEFAULT],
                        nargs=1, help="input excel or csv file")
    parser.add_argument('-c', '--condition', metavar='CONDITION', default=[],
                        nargs='+', help="condition excel or csv file")
    parser.add_argument('--color', metavar='COLOR', default=[''],
                        nargs=1, help="color")
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
    parser.add_argument('--groupstyle', metavar='GROUPSTYLE', default=['box'], type=str,
                        nargs=1, help="grouping style (line or box)")
    parser.add_argument('--grouptextblank', action='store_true')
    parser.add_argument('--chartstyle', metavar='CHARTSTYLE', default=['chart'], type=str,
                        nargs=1, help="chart style (chart or bar)")
    parser.add_argument('--spec', metavar='SPEC',
                        nargs='+', help="add spec indication")
    parser.add_argument('--gui', action='store_true')

    args = vars(parser.parse_args())

    logger.info('parsed input arguments.')

    args['x'] = args['x'][0]
    args['y'] = args['y'][0]
    args['t'] = args['t'][0]
    args['yscale'] = args['yscale'][0]
    args['input'] = args['input'][0]
    if args['output'] is not None:
        args['output'] = args['output'][0]
    args['groupstyle'] = args['groupstyle'][0]
    args['chartstyle'] = args['chartstyle'][0]
    if args['ymax'] is not None:
        args['ymax'] = args['ymax'][0]
    args['color_table'] = dict()
    args['color'] = args['color'][0]

    if args['gui']:

        logger.info('starting gui.')

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
            logger.info('getting working directory.')
            tmp = fd.askdirectory()
            if tmp != '':
                input_dirname.set(op.normpath(tmp))
                tmppf = op.normpath(op.join(tmp, profile_DEFAULT))
                if op.exists(tmppf):
                    input_profile.set(tmppf)
            return

        def get_pf():
            logger.info('getting plot profile.')
            tmp = fd.askopenfilename()
            if tmp != '':
                input_profile.set(op.normpath(tmp))
            return

        def run():
            if input_dirname.get() != '':
                logger.info('starting plotting.')
                if input_input.get() == '':
                    args['input'] = op.normpath(
                        op.join(input_dirname.get(), input_DEFAULT))
                else:
                    args['input'] = op.normpath(
                        op.join(input_dirname.get(), input_input.get()))
                if input_output.get() == '':
                    args['output'] = op.normpath(op.join(
                        input_dirname.get(), output_DEFAULT))
                else:
                    args['output'] = op.normpath(
                        op.join(input_dirname.get(), input_output.get()))
                if not (input_cond0.get() == ''):
                    args['condition'].append(op.normpath(
                        op.join(input_dirname.get(), input_cond0.get())))
                if not (input_cond1.get() == ''):
                    args['condition'].append(op.normpath(
                        op.join(input_dirname.get(), input_cond1.get())))
                if not (input_cond2.get() == ''):
                    args['condition'].append(op.normpath(
                        op.join(input_dirname.get(), input_cond2.get())))
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
                if group_text_disable.get():
                    args['grouptextblank'] = True
                else:
                    args['grouptextblank'] = False
                if spec_enable.get():
                    args['spec'] = []
                    if (spec_name0.get() != '') and (spec_value0.get() != ''):
                        args['spec'].append(spec_name0.get())
                        args['spec'].append(spec_value0.get())
                    if (spec_name1.get() != '') and (spec_value1.get() != ''):
                        args['spec'].append(spec_name1.get())
                        args['spec'].append(spec_value1.get())
            else:
                logger.info(
                    'no working directory set in gui, continuing with cli arguments')

            main(args)
            return

        def load_config():
            try:
                tmp = pd.read_excel(input_profile.get())
                tmp.fillna('', inplace=True)
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
            except:
                return

        def save_config():
            if input_profile.get() == '':
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
            try:
                tmpdf.to_excel(input_profile.get(),
                               sheet_name='Sheet1', index=False)
            except:
                pass

        # row 0
        input_dirname = StringVar()
        input_profile = StringVar()
        input_input = StringVar()
        input_output = StringVar()
        input_cond0 = StringVar()
        input_cond1 = StringVar()
        input_cond2 = StringVar()

        to_save.append(['input_input', 'String'])
        to_save.append(['input_output', 'String'])
        to_save.append(['input_cond0', 'String'])
        to_save.append(['input_cond1', 'String'])
        to_save.append(['input_cond2', 'String'])

        row00 = Frame(row0)
        row01 = Frame(row0)
        row02 = Frame(row0)

        row00.grid(row=0, column=0, sticky='W')
        row01.grid(row=1, column=0, sticky='W')
        row02.grid(row=2, column=0, sticky='W')

        # row00
        input_b1 = Button(row00, width=6, text="folder", command=get_wd)
        input_et1 = Entry(row00, width=61, textvariable=input_dirname)
        input_b0 = Button(row00, width=6, text="run", command=run)
        input_b6 = Button(row00, width=6, text="help")

        input_b1.grid(row=0, column=0)
        input_et1.grid(row=0, column=1)
        input_b0.grid(row=0, column=2)
        input_b6.grid(row=0, column=3)

        # row01
        input_b4 = Button(row01, width=6, text="profile", command=get_pf)
        input_et0 = Entry(row01, width=61, textvariable=input_profile)
        input_b2 = Button(row01, width=6, text="load",
                          command=load_config)
        input_b3 = Button(row01, width=6, text="save",
                          command=save_config)

        input_b4.grid(row=0, column=0)
        input_et0.grid(row=0, column=1)
        input_b2.grid(row=0, column=2)
        input_b3.grid(row=0, column=3)

        # row02
        input_et2 = Entry(row02, width=8, textvariable=input_input)
        input_et3 = Entry(row02, width=8, textvariable=input_output)
        input_et4 = Entry(row02, width=8, textvariable=input_cond0)
        input_et5 = Entry(row02, width=8, textvariable=input_cond1)
        input_et6 = Entry(row02, width=8, textvariable=input_cond2)

        Label(row02, text='data', width=5).grid(row=0, column=0)
        input_et2.grid(row=0, column=1)
        Label(row02, text='chart', width=5).grid(row=0, column=2)
        input_et3.grid(row=0, column=3)
        Label(row02, text='cond_0', width=7).grid(row=0, column=4)
        input_et4.grid(row=0, column=5)
        Label(row02, text='cond_1', width=7).grid(row=0, column=6)
        input_et5.grid(row=0, column=7)
        Label(row02, text='cond_2', width=7).grid(row=0, column=8)
        input_et6.grid(row=0, column=9)

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
        group_text_disable = BooleanVar()
        group_style = IntVar()
        group_type0 = StringVar()
        group_type1 = StringVar()
        group_type2 = StringVar()

        to_save.append(['group_enable', 'Boolean'])
        to_save.append(['group_text_disable', 'Boolean'])
        to_save.append(['group_style', 'Int'])
        to_save.append(['group_type0', 'String'])
        to_save.append(['group_type1', 'String'])
        to_save.append(['group_type2', 'String'])

        group_cb0 = Checkbutton(row3, text="enable", variable=group_enable)
        group_cb1 = Checkbutton(row3, text="hide text",
                                variable=group_text_disable)
        group_rb0 = Radiobutton(
            row3, text="box", variable=group_style, value=0)
        group_rb1 = Radiobutton(
            row3, text="line", variable=group_style, value=1)
        group_et0 = Entry(row3, width=8, textvariable=group_type0)
        group_et1 = Entry(row3, width=8, textvariable=group_type1)
        group_et2 = Entry(row3, width=8, textvariable=group_type2)

        group_cb0.grid(row=0, column=0)
        group_cb1.grid(row=0, column=1)
        group_rb0.grid(row=0, column=2)
        group_rb1.grid(row=0, column=3)
        Label(row3, text='group 1').grid(row=0, column=4)
        group_et0.grid(row=0, column=5)
        Label(row3, text='group 2').grid(row=0, column=6)
        group_et1.grid(row=0, column=7)
        Label(row3, text='group 3').grid(row=0, column=8)
        group_et2.grid(row=0, column=9)

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
