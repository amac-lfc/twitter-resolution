"""
==========
Table Demo
==========

Demo of table function to display a table within a plot.
"""
import numpy as np
import matplotlib.pyplot as plt



PG = np.load('arrays/PG.npy', allow_pickle=True)
H = np.load('arrays/H.npy',allow_pickle=True)
HF = np.load('arrays/HF.npy', allow_pickle=True)
GS = np.load('arrays/GS.npy', allow_pickle=True)
RL = np.load('arrays/RL.npy', allow_pickle=True)
C = np.load('arrays/C.npy', allow_pickle=True)
FF =  np.load('arrays/FF.npy', allow_pickle=True)
F =  np.load('arrays/F.npy', allow_pickle=True)
E =  np.load('arrays/E.npy', allow_pickle=True)
TM = np.load('arrays/TM.npy', allow_pickle=True)
R =  np.load('arrays/R.npy', allow_pickle=True)
D =  np.load('arrays/D.npy', allow_pickle=True)

data = np.array([PG, H, HF, GS, RL, C, F, TM, R, D, FF, E])

# columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
# rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

columns = ('AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC')
rows = ('PG', 'H', 'HF', 'GS', 'RL', 'C', 'F', 'TM', 'R', 'D', 'FF', 'E')

values = np.arange(0, 10,1)
value_increment = 1

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()

# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("# of Tweets".format(value_increment))
plt.yticks(values * value_increment, ['%d' % val for val in values])
plt.xticks([])
plt.title('Loss by Disaster')

plt.show()
