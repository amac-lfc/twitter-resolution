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
data = data.join(followdata)
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
print("len=", len(data))
print("followers=",data.followers)
arrayX = np.zeros([len(data), 3],'i')
for i in range(0, len(data)-2):
    arrayX[i, 0] = data.gender[i+1]
    arrayX[i, 1] = data.tweet_state[i+1]
    arrayX[i, 2] = int(data.followers[i+1])
    # arrayX[i, 2] = data.tweet_region[i]

print("gender, state, followers \n", arrayX)

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
print("this is array Y=", arrayY)



# for i in range(0, len(data)-1):
#     arrayY[i,0] = data.Resolution_Category[i+1]
# print("tweet category\n", arrayY)
# print(len(arrayY))

[chi2,pval] = sk.chi2(arrayX, arrayY)
print("chi2 value for gender, state, followers with category", chi2)
print("these are the pvalues=", pval)

z = np.concatenate((arrayX, arrayY), axis = 1)
print("adjoined=",z)
np.random.shuffle(z)
print("shuffled=",z)

print("80 percent is at", int(0.8*len(z)))
Xtrain = z[:int((.8*len(z))),:3]
Ytrain = z[:int((.8*len(z))),3]
Xtest = z[int(0.8*len(z)):,:3]
Ytest = z[int(0.8*len(z)):,3]

# decision tree
tr = DecisionTreeClassifier()
tr.fit(Xtrain, Ytrain)
print("importances=",tr.feature_importances_)

y_predict = tr.predict(Xtest)
print(f"Accuracy score for Random Forest Classifier is: {accuracy_score(Ytest, y_predict)}")


tr2 = LogisticRegression()
tr2.fit(Xtrain, Ytrain)
# print("importances =", tr2.feature_importances_)

y_predict = tr2.predict(Xtest)
print(f"Accuracy score for Logistic Regression Classifier is: {accuracy_score(Ytest, y_predict)}")

# histogram
# geocount = data.pivot_table(index=['tweet_state'], aggfunc='size')
# print(geocount)
# hist = geocount.plot.hist(grid=True, bins=20, rwidth=0.9,
                   # color='#607c8e')

# matplotlib histogram
# for i in range(0,len(data)):
arr  = []
for i in range(0, len(data)-1):
     arr.append(data.tweet_state[i+1])
# print(arr)
n, bins, patches = plt.hist(x = arr, bins = np.arange(0, 51, step = 1), color = '#A737B5',
                            alpha = 0.7, rwidth = 0.8)

plt.grid(axis= 'y', alpha = 0.75)
plt.xlabel('State')
plt.ylabel('frequency')
plt.xticks(np.arange(0, 51, step = 1))
plt.title("Tweet Frequency by State")
labels = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN',
            'IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','NE','NV','NH',
            'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI', 'SC','SD','TN','TX',
            'UT','VT', 'VA','WA','WV','WI','WY','DC']
plt.xticks(np.arange(0, 51, step = 1), labels)
plt.show()

#divde the data by gender
arrW = np.zeros(51, dtype = int) #for women
arrM = np.zeros(51, dtype = int) #for men
for i in range(0, len(data)-1):
    if data.gender[i] == 0:
        x = data.tweet_state[i]
        arrW[x] = arrW[x] + 1
    if data.gender[i] == 1:
        x = data.tweet_state[i]
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


print(data['Good Samaritan '][0:10])
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
            'IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','NE','NV','NH',
            'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI', 'SC','SD','TN','TX',
            'UT','VT', 'VA','WA','WV','WI','WY','DC']
plt.xticks(np.arange(0, 51, step = 1), labels1)
plt.legend((p0[0],p1[0],p2[0],p3[0],p4[0],p5[0],p6[0],p7[0],p8[0], p9[0], p10[0], p11[0]), ('Health & Fitness', 'Humor', 'Personal Growth', 'Philanthropic', 'Recreation & Leisure', 'Career','Family/Friends/Relationships', 'Finance', 'Education/Training', 'Time Management/Organization' ), bbox_to_anchor=(1.25, 1), loc = 'upper right', borderaxespad = 0.)
plt.title("Tweet Frequency by State and Category")
plt.show()


#trying to do a plot with a table connected to the bottom
# categories = np.concatenate((HF, H, PG, P, RL, C, FF, F, ET, TM))
categories = np.array([PG, H, HF, GS, RL, C, F, TM, R, D, FF, E])# states = categories.transpose()
print("categories matrix=",categories)
# print("HF=", HF)
# print("new=",categories[1,])
columns = ('AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC')
rows = ('PG', 'H', 'HF', 'GS', 'RL', 'C', 'F', 'TM', 'R', 'D', 'FF', 'E')

values = np.arange(0, 1000, 100)
value_increment = 2000

colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(categories)
# print("first=", categories[1])

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# initialize the vertical-offset for the stacked bar chart
y_offset = np.zeros(len(columns))

#Plot bars and create text lables for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, categories[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + categories[row]
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

plt.ylabel("Tweet Frequency".format(value_increment))
plt.yticks(values * value_increment, ['%d' % val for val in values])
plt.xticks([])
plt.title('Tweets by State')

plt.show()
