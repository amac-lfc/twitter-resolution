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
# from yellowbrick.classifier import ConfusionMatrix
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")
# equivalents from matplotlib
# data = pd.read_csv("dfe.csv",encoding = "ISO-8859-1")
# print(data[data.columns[1]])
data = pd.read_excel('dfe2.xlsx')
# print("column headings:")
# print(data.columns.values)

arrayX = np.load('arrays/X.npy', allow_pickle=True)
arrayY = np.load('arrays/Y.npy',allow_pickle=True)
# arrayY = np.load('arrays/only_pg.npy', allow_pickle=True)
# arrayY = np.transpose(arrayY)
arrayY[arrayY!=0]=1
print(arrayX, arrayY)
print(arrayX.shape, arrayY.shape)

# [chi2,pval] = sk.chi2(arrayX, arrayY)
# print("chi2 value for gender, state, followers with category", chi2)
# print("these are the pvalues=", pval)

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
# tr2.fit(Xtrain, Ytrain)
# print("importances =", tr2.feature_importances_)

# y_predict = tr2.predict(Xtest)
# print(f"Accuracy score for Logistic Regression Classifier is: {accuracy_score(Ytest, y_predict)}")
# print("importances=",tr2.coef_)

#confusion ConfusionMatrix only commented bc lab doesnt have yellow brick
# cm = ConfusionMatrix(tr2, classes=[0,1])
# cm.fit(Xtrain, Ytrain)
# cm.score(Xtest, Ytest)
# cm.show()

# SMOTE

c1, c2 = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                             n_informative=3, n_redundant=1, flip_y=0,
                             n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
print('Original dataset shape %s' % Counter(c2))
sm = SMOTE(random_state=42)
c1_res, c2_res = sm.fit_resample(c1, c2)
print('Resampled dataset shape %s' % Counter(c2_res))
