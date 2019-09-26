import pandas as pd
import numpy as np
import sklearn.feature_selection as sk
import scipy.stats as sp
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from example import *
import matplotlib.pyplot as plt
# equivalents from matplotlib


data = pd.read_csv("dfe.csv",encoding = "ISO-8859-1")
print(data[data.columns[1]])

data.Resolution_Category[data.Resolution_Category == 'Health & Fitness'] = 0
data.Resolution_Category[data.Resolution_Category == 'Humor'] = 1
data.Resolution_Category[data.Resolution_Category == 'Personal Growth'] = 2
data.Resolution_Category[data.Resolution_Category == 'Philanthropic'] = 3
data.Resolution_Category[data.Resolution_Category == 'Recreation & Leisure'] = 4
data.Resolution_Category[data.Resolution_Category == 'Career'] = 5
data.Resolution_Category[data.Resolution_Category == 'Family/Friends/Relationships'] = 6
data.Resolution_Category[data.Resolution_Category == 'Finance'] = 7
data.Resolution_Category[data.Resolution_Category == 'Education/Training'] = 8
data.Resolution_Category[data.Resolution_Category == 'Time Management/Organization'] = 9

print(data[data.columns[4]])

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

arrayX = np.zeros([len(data), 2])
for i in range(0, len(data)-1):
    arrayX[i, 0] = data.gender[i+1]
    arrayX[i, 1] = data.tweet_state[i+1]
    # arrayX[i, 2] = data.tweet_region[i]

print("gender, state, region \n", arrayX)

arrayY = np.zeros([len(data), 1])
for i in range(0, len(data)-1):
    arrayY[i,0] = data.Resolution_Category[i+1]
print("tweet category\n", arrayY)
# print(len(arrayY))

[chi2,pval] = sk.chi2(arrayX, arrayY)
print(chi2)
print(pval)

z = np.concatenate((arrayX, arrayY), axis = 1)
print("adjoined=",z)
np.random.shuffle(z)
print("shuffled=",z)

print("80 percent is at", int(0.8*len(z)))
Xtrain = z[:int((.8*len(z))),:2]
Ytrain = z[:int((.8*len(z))),2]
Xtest = z[int(0.8*len(z)):,:2]
Ytest = z[int(0.8*len(z)):,2]
# print(atrain)

# decision tree
tr = DecisionTreeClassifier()
tr.fit(Xtrain, Ytrain)

y_predict = tr.predict(Xtest)
print(f"Accuracy score for Random Forest Classifier is: {accuracy_score(Ytest, y_predict)}")

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
n, bins, patches = plt.hist(x = arr, bins = np.arange(0, 51, step = 1), color = '#000000',
                            alpha = 0.7, rwidth = 0.8)
plt.grid(axis= 'y', alpha = 0.75)
plt.xlabel('State')
plt.ylabel('frequency')
plt.xticks(np.arange(0, 51, step = 1))
plt.title("Tweet Frequency by State")
# plt.show()

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

# stacked bar plot for category
HF = np.zeros(51, dtype = int) #for 'Health & Fitness'
H = np.zeros(51, dtype = int) #for 'Humor'
PG = np.zeros(51, dtype = int) #for 'Personal Growth'
P = np.zeros(51, dtype = int) #for 'Philanthropic'
RL = np.zeros(51, dtype = int) #for 'Recreation and Leisure'
C = np.zeros(51, dtype = int) #for 'Career'
FF = np.zeros(51, dtype = int) #for 'Family/Friends'
F = np.zeros(51, dtype = int) #for 'Finance'
ET = np.zeros(51, dtype = int) #for 'Education/Training'
TM = np.zeros(51, dtype = int) #for 'Time Management'

# convert the categories to integer values
for i in range(0, len(data)-1):
    if data.Resolution_Category[i] == 0:
        x = data.tweet_state[i]
        HF[x] = HF[x] + 1
    if data.Resolution_Category[i] == 1:
        x = data.tweet_state[i]
        H[x] = H[x] + 1
    if data.Resolution_Category[i] == 2:
        x = data.tweet_state[i]
        PG[x] = PG[x] + 1
    if data.Resolution_Category[i] == 3:
        x = data.tweet_state[i]
        P[x] = P[x] + 1
    if data.Resolution_Category[i] == 4:
        x = data.tweet_state[i]
        RL[x] = RL[x] + 1
    if data.Resolution_Category[i] == 5:
        x = data.tweet_state[i]
        C[x] = C[x] + 1
    if data.Resolution_Category[i] == 6:
        x = data.tweet_state[i]
        FF[x] = FF[x] + 1
    if data.Resolution_Category[i] == 7:
        x = data.tweet_state[i]
        F[x] = F[x] + 1
    if data.Resolution_Category[i] == 8:
        x = data.tweet_state[i]
        ET[x] = ET[x] + 1
    if data.Resolution_Category[i] == 9:
        x = data.tweet_state[i]
        TM[x] = TM[x] + 1
# print("HF =", HF)
# print("m =", arrM)
width = 0.5
ind = np.arange(51)
p0 = plt.bar(ind, HF, width)
p1 = plt.bar(ind, H, width, bottom = HF)
p2 = plt.bar(ind, PG, width, bottom = HF + H)
p3 = plt.bar(ind, P, width, bottom = HF + H + PG)
p4 = plt.bar(ind, RL, width, bottom = HF + H + PG + P)
p5 = plt.bar(ind, C, width, bottom = HF + H + PG + P + RL)
p6 = plt.bar(ind, FF, width, bottom = HF + H + PG + P + RL + C)
p7 = plt.bar(ind, F, width, bottom = HF + H + PG + P + RL + C + FF)
p8 = plt.bar(ind, ET, width, bottom = HF + H + PG + P + RL + C + FF + F)
p9 = plt.bar(ind, TM, width, bottom = HF + H + PG + P + RL + C + FF + F + ET)

plt.ylabel('Tweet Frequency')
plt.xlabel('State')
plt.xticks(np.arange(0, 51, step = 1))
plt.legend((p0[0],p1[0],p2[0],p3[0],p4[0],p5[0],p6[0],p7[0],p8[0], p9[0]), ('Health & Fitness', 'Humor', 'Personal Growth', 'Philanthropic', 'Recreation & Leisure', 'Career','Family/Friends/Relationships', 'Finance', 'Education/Training', 'Time Management/Organization' ), bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad = 0.)
plt.title("Tweet Frequency by State and Category")
# plt.show()

## here you are adding the number of followers to the excel
CONSUMER_KEY = 'n4qNDyH25RLTRW1wLlKa4lLU6'
CONSUMER_SECRET = 'OIN2Vi1Wdq8mP8QLFaiN6sGvc2LeVHLdJMECVEiyrUk9j8z9zD'
ACCESS_TOKEN = '1171636288086315009-3uesxkcQ5TKbijuUaITb05yOr8fu51'
ACCESS_TOKEN_SECRET = 'aY654202Zk9LCJ01x9zcTfqarMpbybp5we5cbKUDZQxEG'
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

names = ["Reesesking1","Stacyloohoo"]

def followercount(names):
    for name in names:
        try:
            user = api.get_user(name)
            ageTwitterAccount = user.created_at
            followers_count =  user.followers_count
            friends_count = user.friends_count
            timezone = user.time_zone
            print(name)
            print(ageTwitterAccount)
            print(followers_count)
            print(friends_count)
            print(timezone)
            return followers_count, friends_count
        except:
            return 0,0


followerlist = []
friendlist = []
name = []

for i in range(0, len(data)-1):
    name.append(data.name[i + 1])
(', '.join('"' + i + '"' for i in name))
# print(name)

for i in range(0, len(name)-1):
    followers, friends = followercount(name)
    followerlist.append(followers)
    friendlist.append(friends)
# error: it says that the user is not found -- maybe they
# deleted their account? what do I do now



#trying to do a plot with a table connected to the bottom
# categories = np.concatenate((HF, H, PG, P, RL, C, FF, F, ET, TM))
categories = [HF, H, PG, P, RL, C, FF, F, ET, TM]
# states = categories.transpose()
print("HF=",HF)
# print("new=",categories[1,])
columns = ('AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY')
rows = ('HF', 'H', 'PG', 'P', 'RL', 'C', 'FF', 'F', 'ET', 'TM')

values = np.arange(0, 10000, 1000)
value_increment = 10000

colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# initialize  the vertical-offset for the stacked bar chart
y_offset = np.zeros(len(columns))

#Plot bars and create text lables for the table
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

plt.ylabel("Loss in ${0}'s".format(value_increment))
plt.yticks(values * value_increment, ['%d' % val for val in values])
plt.xticks([])
plt.title('Loss by Disaster')

# plt.show()
