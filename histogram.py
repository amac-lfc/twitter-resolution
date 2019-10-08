import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import sklearn.feature_selection as sk
import scipy.stats as sp
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

X = np.load('arrays/X.npy', allow_pickle=True)
Y = np.load('arrays/Y.npy',allow_pickle=True)


arr  = X[:,1]
print(arr)
n, bins, patches = plt.hist(x = arr, bins = np.arange(0, 51, step = 1), color = '#A737B5',
                            alpha = 0.7, rwidth = 0.8)

plt.grid(axis= 'y', alpha = 0.75)
plt.xlabel('State')
plt.ylabel('frequency')
# plt.xticks(np.arange(0, 51, step = 1))
plt.title("Tweet Frequency by State")
labels = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN',
            'IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH',
            'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI', 'SC','SD','TN','TX',
            'UT','VT', 'VA','WA','WV','WI','WY','DC']
plt.xticks(np.arange(0, 51, step = 1), labels)
plt.show()

#divde the data by gender
arrW = np.zeros(51, dtype = int) #for women
arrM = np.zeros(51, dtype = int) #for men
for i in range(0, len(X)):
    if X[i,0] == 0:
        x = X[i,1]
        arrW[x] = arrW[x] + 1
    if X[i,0] == 1:
        x = X[i,1]
        arrM[x] = arrM[x] + 1
# print("w =", arrW)
# print("m =", arrM)
width = 0.5
ind = np.arange(51)
p1 = plt.bar(ind, arrM, width)
p2 = plt.bar(ind, arrW, width, bottom = arrM)
plt.ylabel('Tweet Frequency')
plt.xlabel('State')
plt.xticks(np.arange(0, 51, step = 1))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))
plt.title("Tweet Frequency by State and Gender")
# plt.show()

# # stacked bar plot for category

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



# # print("HF =", HF)
# # print("m =", arrM)
width = 0.5
ind = np.arange(51)
p0 = plt.bar(ind, PG, width)
p1 = plt.bar(ind, H, width, bottom = PG)
p2 = plt.bar(ind, HF, width, bottom = PG + H)
p3 = plt.bar(ind, GS, width, bottom = PG + H + HF)
p4 = plt.bar(ind, RL, width, bottom = PG + H + HF + GS)
p5 = plt.bar(ind, C, width, bottom = PG + H + HF + GS + RL)
p6 = plt.bar(ind, F, width, bottom = PG + H + HF + GS + RL + C)
p7 = plt.bar(ind, TM, width, bottom = PG + H + HF + GS + RL + C + F)
p8 = plt.bar(ind, R, width, bottom = PG + H + HF + GS + RL + C + F + TM)
p9 = plt.bar(ind, D, width, bottom = PG + H + HF + GS + RL + C + F + TM + R)
p10 = plt.bar(ind, FF, width, bottom = PG + H + HF + GS + RL + C + F + TM + R + D)
p11 = plt.bar(ind, E, width, bottom =  PG + H + HF + GS + RL + C + F + TM + R + D + FF)


plt.ylabel('Tweet Frequency')
plt.xlabel('State')
labels1 = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN',
            'IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH',
            'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI', 'SC','SD','TN','TX',
            'UT','VT', 'VA','WA','WV','WI','WY','DC']
plt.xticks(np.arange(0, 51, step = 1), labels1)
plt.legend((p0[0],p1[0],p2[0],p3[0],p4[0],p5[0],p6[0],p7[0],p8[0], p9[0], p10[0], p11[0]), ('Health & Fitness', 'Humor', 'Personal Growth', 'Philanthropic', 'Recreation & Leisure', 'Career','Family/Friends/Relationships', 'Finance', 'Education/Training', 'Time Management/Organization' ), bbox_to_anchor=(1.25, 1), loc = 'upper right', borderaxespad = 0.)
plt.title("Tweet Frequency by State and Category")
plt.show()
