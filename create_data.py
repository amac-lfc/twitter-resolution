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
# equivalents from matplotlib
# data = pd.read_csv("dfe.csv",encoding = "ISO-8859-1")
# print(data[data.columns[1]])
data = pd.read_excel('dfe2.xlsx')
# print("column headings:")
# print(data.columns.values)

followdata = pd.read_csv("followers.csv")
print(followdata.columns.values)
print("data = ", data.columns.values)



# code the state strings into integers
data.tweet_state[data.tweet_state == 'AL'] = 0
data.tweet_state[data.tweet_state == 'AK'] = 1
data.tweet_state[data.tweet_state == 'AZ'] = 2
data.tweet_state[data.tweet_state == 'AR'] = 3
data.tweet_state[data.tweet_state == 'CA'] = 4
data.tweet_state[data.tweet_state == 'CO'] = 5
data.tweet_state[data.tweet_state == 'CT'] = 6
data.tweet_state[data.tweet_state == 'DE'] = 7
data.tweet_state[data.tweet_state == 'FL'] = 8
data.tweet_state[data.tweet_state == 'GA'] = 9
data.tweet_state[data.tweet_state == 'HI'] = 10
data.tweet_state[data.tweet_state == 'ID'] = 11
data.tweet_state[data.tweet_state == 'IL'] = 12
data.tweet_state[data.tweet_state == 'IN'] = 13
data.tweet_state[data.tweet_state == 'IA'] = 14
data.tweet_state[data.tweet_state == 'KS'] = 15
data.tweet_state[data.tweet_state == 'KY'] = 16
data.tweet_state[data.tweet_state == 'LA'] = 17
data.tweet_state[data.tweet_state == 'ME'] = 18
data.tweet_state[data.tweet_state == 'MD'] = 19
data.tweet_state[data.tweet_state == 'MA'] = 20
data.tweet_state[data.tweet_state == 'MI'] = 21
data.tweet_state[data.tweet_state == 'MN'] = 22
data.tweet_state[data.tweet_state == 'MS'] = 23
data.tweet_state[data.tweet_state == 'MO'] = 24
data.tweet_state[data.tweet_state == 'MT'] = 25
data.tweet_state[data.tweet_state == 'NE'] = 26
data.tweet_state[data.tweet_state == 'NV'] = 27
data.tweet_state[data.tweet_state == 'NH'] = 28
data.tweet_state[data.tweet_state == 'NJ'] = 29
data.tweet_state[data.tweet_state == 'NM'] = 30
data.tweet_state[data.tweet_state == 'NY'] = 31
data.tweet_state[data.tweet_state == 'NC'] = 32
data.tweet_state[data.tweet_state == 'ND'] = 33
data.tweet_state[data.tweet_state == 'OH'] = 34
data.tweet_state[data.tweet_state == 'OK'] = 35
data.tweet_state[data.tweet_state == 'OR'] = 36
data.tweet_state[data.tweet_state == 'PA'] = 37
data.tweet_state[data.tweet_state == 'RI'] = 38
data.tweet_state[data.tweet_state == 'SC'] = 39
data.tweet_state[data.tweet_state == 'SD'] = 40
data.tweet_state[data.tweet_state == 'TN'] = 41
data.tweet_state[data.tweet_state == 'TX'] = 42
data.tweet_state[data.tweet_state == 'UT'] = 43
data.tweet_state[data.tweet_state == 'VT'] = 44
data.tweet_state[data.tweet_state == 'VA'] = 45
data.tweet_state[data.tweet_state == 'WA'] = 46
data.tweet_state[data.tweet_state == 'WV'] = 47
data.tweet_state[data.tweet_state == 'WI'] = 48
data.tweet_state[data.tweet_state == 'WY'] = 49
data.tweet_state[data.tweet_state == 'DC'] = 50

data.gender[data.gender == 'female'] = 0
data.gender[data.gender == 'male'] = 1

data.tweet_region[data.tweet_region == 'West'] = 0
data.tweet_region[data.tweet_region == 'Northeast'] = 1
data.tweet_region[data.tweet_region == 'South'] = 2
data.tweet_region[data.tweet_region == 'Midwest'] = 3

# move into a numpy array
# i; tweet
# y[i]; category
# x[i, 0]; 0, 1 gender
# x[i, 1]; where they tweeted 0, 49 states ,no more states
# x[i, 2]; user time zone 0 to 23, no more time zone
# x[i, 3]; tweet region
# chi square test
# PCA test
# reading about decision trees
arrayX = np.zeros([len(data), 3],'i')
for i in range(0, len(data)-2):
    arrayX[i, 0] = data.gender[i+1]
    arrayX[i, 1] = data.tweet_state[i+1]
    arrayX[i, 2] = int(followdata.followers[i+1])
    # arrayX[i, 2] = data.tweet_region[i]

# print("gender, state, followers \n", arrayX)

arrayY = np.zeros([len(data), 1], 'i')

for i in range(0, len(data)-1):
     if data['Personal Growth'][i] == 1:
         arrayY[i, 0] = 0
     elif data['Humor'][i] == 1:
         arrayY[i, 0] = 1
     elif data['Health and Fitness'][i] == 1:
         arrayY[i, 0] = 2
     elif data['Good Samaritan '][i] == 1:
         arrayY[i, 0] = 3
     elif data['Rec and Leisure'][i] == 1:
         arrayY[i, 0] = 4
     elif data['Career'][i] == 1:
         arrayY[i, 0] = 5
     elif data['Finance'][i] == 1:
         arrayY[i, 0] = 6
     elif data['Time Management'][i] == 1:
         arrayY[i, 0] = 7
     elif data['Religion'][i] == 1:
         arrayY[i, 0] = 8
     elif data['Dating'][i] == 1:
         arrayY[i, 0] = 9
     elif data['Friends and Family'][i] == 1:
         arrayY[i, 0] = 10
     elif data['Education'][i] == 1:
         arrayY[i, 0] = 11
     elif data['Omit'][i] == 1:
        arrayY[i, 0] = 12
# print("this is array Y=", arrayY)

np.save('arrays/X', arrayX,allow_pickle=True)
np.save('arrays/Y', arrayY,allow_pickle=True)


# # stacked bar plot for category
PG = np.zeros(51, dtype = int) #for 'Personal Growth'
H = np.zeros(51, dtype = int) #for 'Humor'
HF = np.zeros(51, dtype = int) #for 'Health & Fitness'
GS = np.zeros(51, dtype = int)
RL = np.zeros(51, dtype = int)
P = np.zeros(51, dtype = int) #for 'Philanthropic'
RL = np.zeros(51, dtype = int) #for 'Recreation and Leisure'
C = np.zeros(51, dtype = int) #for 'Career'
FF = np.zeros(51, dtype = int) #for 'Family/Friends'
F = np.zeros(51, dtype = int) #for 'Finance'
E = np.zeros(51, dtype = int) #for 'Education/Training'
TM = np.zeros(51, dtype = int) #for 'Time Management'
R = np.zeros(51, dtype = int) #for 'Time Management'
D = np.zeros(51, dtype = int) #for 'Time Management'


# print(data['Good Samaritan '][0:10])
# convert the categories to integer values
for i in range(0, len(data)-1):
     if data['Personal Growth'][i] == 1:
         x = data.tweet_state[i]
         PG[x] = PG[x] + 1
     if data['Humor'][i] == 1:
         x = data.tweet_state[i]
         H[x] = H[x] + 1
     if data['Health and Fitness'][i] == 1:
         x = data.tweet_state[i]
         HF[x] = HF[x] + 1
     if data['Good Samaritan '][i] == 1:
         x = data.tweet_state[i]
         GS[x] = GS[x] + 1
     if data['Rec and Leisure'][i] == 1:
         x = data.tweet_state[i]
         RL[x] = RL[x] + 1
     if data['Career'][i] == 1:
         x = data.tweet_state[i]
         C[x] = C[x] + 1
     if data['Finance'][i] == 1:
         x = data.tweet_state[i]
         F[x] = F[x] + 1
     if data['Time Management'][i] == 1:
         x = data.tweet_state[i]
         TM[x] = TM[x] + 1
     if data['Religion'][i] == 1:
         x = data.tweet_state[i]
         R[x] = R[x] + 1
     if data['Dating'][i] == 1:
         x = data.tweet_state[i]
         D[x] = D[x] + 1
     if data['Friends and Family'][i] == 1:
         x = data.tweet_state[i]
         FF[x] = FF[x] + 1
     if data['Education'][i] == 1:
         x = data.tweet_state[i]
         E[x] = E[x] + 1

np.save('arrays/PG', PG, allow_pickle=True)
np.save('arrays/H', H, allow_pickle=True )
np.save('arrays/HF', HF, allow_pickle=True)
np.save('arrays/GS', GS, allow_pickle=True)
np.save('arrays/RL', RL, allow_pickle=True)
np.save('arrays/C', C, allow_pickle=True)
np.save('arrays/FF', FF, allow_pickle=True)
np.save('arrays/F', F, allow_pickle=True)
np.save('arrays/E', E, allow_pickle=True)
np.save('arrays/TM', TM, allow_pickle=True)
np.save('arrays/R', R, allow_pickle=True)
np.save('arrays/D', D, allow_pickle=True)


only_pg = data['Personal Growth'].as_matrix()
only_h = data['Humor'].as_matrix()
only_hf = data['Health and Fitness'].as_matrix()
only_gs = data['Good Samaritan '].as_matrix()
only_rl = data['Rec and Leisure'].as_matrix()
only_c = data['Career'].as_matrix()
only_ff = data['Friends and Family'].as_matrix()
only_f = data['Finance'].as_matrix()
only_e = data['Education'].as_matrix()
only_tm = data['Time Management'].as_matrix()
only_r = data['Religion'].as_matrix()
only_d = data['Dating'].as_matrix()
print(only_hf)
print(only_h)

np.save('arrays/only_pg', only_pg, allow_pickle=True)
np.save('arrays/only_h', only_h, allow_pickle=True )
np.save('arrays/only_hf', only_hf, allow_pickle=True)
np.save('arrays/only_gs', only_gs, allow_pickle=True)
np.save('arrays/only_rl', only_rl, allow_pickle=True)
np.save('arrays/only_c', only_c, allow_pickle=True)
np.save('arrays/only_ff', only_ff, allow_pickle=True)
np.save('arrays/only_f', only_f, allow_pickle=True)
np.save('arrays/only_e', only_e, allow_pickle=True)
np.save('arrays/only_tm', only_tm, allow_pickle=True)
np.save('arrays/only_r', only_r, allow_pickle=True)
np.save('arrays/only_d', only_d, allow_pickle=True)
