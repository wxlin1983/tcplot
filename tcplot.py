# import win32com.client as win32

# excel = win32.gencache.EnsureDispatch('Excel.Application')
# wb = excel.Workbooks.Open('C:\\Users\\weixu\\Downloads\\pyexcel\\workbook1.xlsx')
# excel.Visible = True
# ws = wb.Worksheets("Sheet1")
# data=ws.Range("A1:E8").Value
# excel.Application.Quit()

import pandas as pd
import numpy as np

df = pd.read_excel(
    'C:\\Users\\weixu\\Downloads\\pyexcel\\workbook1.xlsx', sheet_name='Sheet1')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib


matplotlib.rc('lines', linewidth=2, color='r')
plt.clf()
plt.close()

fig_size_x = 10
fig_size_y = 2.5
fig_dpi = 300

ax_offset_x = 0.05
ax_offset_y = 0.45
ax_size_x = 0.925
ax_size_y = 0.5
# rc('axes', linewidth=2)

fig = plt.figure(figsize=[fig_size_x, fig_size_y], dpi=fig_dpi)
ax = fig.add_axes([ax_offset_x, ax_offset_y, ax_size_x, ax_size_y])

od = list(range(1, len(df.id.tolist()) + 1))
ax.plot(od, df.value, linewidth=2.0, zorder=1)
# ax1.plot(df.id,df.value)
plt.xticks(od, df.id.tolist(), rotation=-90)
ax.tick_params(labelsize=6)
ax.set_xlim([0.5, 7.5])
ax.set_ylim([0, 4])
# [i.set_linewidth(0.1) for i in ax.spines.itervalues()]
# fontsize = 14
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

# ax.axhline(linewidth=4, color="g")        # inc. width of x-axis and color it green
# ax.axvline(linewidth=4, color="r")        # inc. width of y-axis and color it red

# for tick in ax.xaxis.get_major_ticks():
#     tick.label1.set_fontsize(fontsize)
#     tick.label1.set_fontweight('bold')

plt.savefig('123.png')

# fig1 = plt.figure(figsize=[8,2])
# ax1 = fig1.add_axes([0.1,0.1,0.5,0.5])
# ax1.add_patch(
#     patches.Rectangle(
#         (-0.05, 0.1),   # (x,y)
#         0.5,          # width
#         0.5,          # height
#         clip_on=False,
#     )
# )
# plt.show()
