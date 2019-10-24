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


for i in range(len(followdata)-1):
    if followdata['created in'][i+1] != 0:
        followdata['created in'][i+1] = followdata['created in'][i+1][:4]


arrayX = np.zeros([len(data), 4],'i')
for i in range(0, len(data)-2):
    arrayX[i, 0] = data.gender[i+1]
    arrayX[i, 1] = data.tweet_state[i+1]
    arrayX[i, 2] = int(followdata.followers[i+1])
    arrayX[i, 3] = int(followdata['created in'][i+1])



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


only_pg = data['Personal Growth'].values
only_h = data['Humor'].values
only_hf = data['Health and Fitness'].values
only_gs = data['Good Samaritan '].values
only_rl = data['Rec and Leisure'].values
only_c = data['Career'].values
only_ff = data['Friends and Family'].values
only_f = data['Finance'].values
only_e = data['Education'].values
only_tm = data['Time Management'].values
only_r = data['Religion'].values
only_d = data['Dating'].values
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

# All language attributes in X and gender in Y ######################333
langdata = pd.read_excel('LIWC2015Results.xlsx')

LX = np.zeros([len(langdata), 90],'i')

for i in range(0, len(langdata)-2):
    LX[i, 0] = langdata.WC[i+1]
    LX[i, 1] = langdata.Analytic[i+1]
    LX[i, 2] = langdata.Clout[i+1]
    LX[i, 3] = langdata.Authentic[i+1]
    LX[i, 4] = langdata.Tone[i+1]
    LX[i, 5] = langdata.function[i+1]
    LX[i, 6] = langdata.pronoun[i+1]
    LX[i, 7] = langdata.ppron[i+1]
    LX[i, 8] = langdata.i[i+1]
    LX[i, 9] = langdata.we[i+1]
    LX[i, 10] = langdata.you[i+1]
    LX[i, 11] = langdata.shehe[i+1]
    LX[i, 12] = langdata.they[i+1]
    LX[i, 13] = langdata.ipron[i+1]
    LX[i, 14] = langdata.article[i+1]
    LX[i, 15] = langdata.prep[i+1]
    LX[i, 16] = langdata.auxverb[i+1]
    LX[i, 17] = langdata.adverb[i+1]
    LX[i, 18] = langdata.conj[i+1]
    LX[i, 19] = langdata.negate[i+1]
    LX[i, 20] = langdata.verb[i+1]
    LX[i, 21] = langdata.adj[i+1]
    LX[i, 22] = langdata.compare[i+1]
    LX[i, 23] = langdata.interrog[i+1]
    LX[i, 24] = langdata.number[i+1]
    LX[i, 25] = langdata.quant[i+1]
    LX[i, 26] = langdata.affect[i+1]
    LX[i, 27] = langdata.posemo[i+1]
    LX[i, 28] = langdata.negemo[i+1]
    LX[i, 29] = langdata.anx[i+1]
    LX[i, 30] = langdata.anger[i+1]
    LX[i, 31] = langdata.sad[i+1]
    LX[i, 32] = langdata.social[i+1]
    LX[i, 33] = langdata.family[i+1]
    LX[i, 34] = langdata.friend[i+1]
    LX[i, 35] = langdata.female[i+1]
    LX[i, 36] = langdata.male[i+1]
    LX[i, 37] = langdata.cogproc[i+1]
    LX[i, 38] = langdata.insight[i+1]
    LX[i, 39] = langdata.cause[i+1]
    LX[i, 40] = langdata.discrep[i+1]
    LX[i, 41] = langdata.tentat[i+1]
    LX[i, 42] = langdata.certain[i+1]
    LX[i, 43] = langdata.differ[i+1]
    LX[i, 44] = langdata.percept[i+1]
    LX[i, 45] = langdata.see[i+1]
    LX[i, 46] = langdata.hear[i+1]
    LX[i, 47] = langdata.feel[i+1]
    LX[i, 48] = langdata.bio[i+1]
    LX[i, 49] = langdata.body[i+1]
    LX[i, 50] = langdata.health[i+1]
    LX[i, 51] = langdata.sexual[i+1]
    LX[i, 52] = langdata.ingest[i+1]
    LX[i, 53] = langdata.drives[i+1]
    LX[i, 54] = langdata.affiliation[i+1]
    LX[i, 55] = langdata.achieve[i+1]
    LX[i, 56] = langdata.power[i+1]
    LX[i, 57] = langdata.reward[i+1]
    LX[i, 58] = langdata.risk[i+1]
    LX[i, 59] = langdata.focuspast[i+1]
    LX[i, 60] = langdata.focuspresent[i+1]
    LX[i, 61] = langdata.focusfuture[i+1]
    LX[i, 62] = langdata.relativ[i+1]
    LX[i, 63] = langdata.motion[i+1]
    LX[i, 64] = langdata.space[i+1]
    LX[i, 65] = langdata.time[i+1]
    LX[i, 66] = langdata.work[i+1]
    LX[i, 67] = langdata.leisure[i+1]
    LX[i, 68] = langdata.home[i+1]
    LX[i, 69] = langdata.money[i+1]
    LX[i, 70] = langdata.relig[i+1]
    LX[i, 71] = langdata.death[i+1]
    LX[i, 72] = langdata.informal[i+1]
    LX[i, 73] = langdata.swear[i+1]
    LX[i, 74] = langdata.netspeak[i+1]
    LX[i, 75] = langdata.assent[i+1]
    LX[i, 76] = langdata.nonflu[i+1]
    LX[i, 77] = langdata.filler[i+1]
    LX[i, 78] = langdata.AllPunc[i+1]
    LX[i, 79] = langdata.Period[i+1]
    LX[i, 80] = langdata.Comma[i+1]
    LX[i, 81] = langdata.Colon[i+1]
    LX[i, 82] = langdata.SemiC[i+1]
    LX[i, 83] = langdata.QMark[i+1]
    LX[i, 84] = langdata.Exclam[i+1]
    LX[i, 85] = langdata.Dash[i+1]
    LX[i, 86] = langdata.Quote[i+1]
    LX[i, 87] = langdata.Apostro[i+1]
    LX[i, 88] = langdata.Parenth[i+1]
    LX[i, 89] = langdata.OtherP[i+1]

LY = np.zeros([len(langdata)],'i')
for i in range(0, len(langdata)-2):
    if langdata.gender[i] == 'female':
        LY[i] = 0
    if langdata.gender[i] == 'male':
        LY[i] = 1


np.save('arrays/LX', LX,allow_pickle=True)
np.save('arrays/LY', LY,allow_pickle=True)
