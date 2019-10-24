""" Boxplots"""
import numpy as np
import matplotlib.pyplot as plt
# do PG read x and y

x = np.load('arrays/X.npy', allow_pickle=True)
y = np.load('arrays/Y.npy',allow_pickle=True)

topic = 0
index = np.where(y==topic)
index2 = np.where(y!=topic)
x1 = x[index, 2]
x2 = x[index2, 2]
print(x1[0], x2[0])
x1 = x1[0]
x2 = x2[0]

# plt.title('Basic Plot')
# box_plot_data = [x1,x2]
# plt.boxplot(box_plot_data, patch_artist=True, labels=['topic', 'nottopic'])
# plt.show()
value1 = [82,76,24,40,67,62,75,78,71,32,98,89,78,67,72,82,87,66,56,52]
value2=[62,5,91,25,36,32,96,95,3,90,95,32,27,55,100,15,71,11,37,21]
value3=[23,89,12,78,72,89,25,69,68,86,19,49,15,16,16,75,65,31,25,52]
value4=[59,73,70,16,81,61,88,98,10,87,29,72,16,23,72,88,78,99,75,30]

box_plot_data=[x1[np.nonzero(x1)], x2[np.nonzero(x2)]]
plt.boxplot(box_plot_data)
plt.show()
