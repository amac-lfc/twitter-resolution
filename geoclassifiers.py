""" tissue data in the brain """

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
from yellowbrick.classifier import ConfusionMatrix
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from imblearn.over_sampling import SMOTE
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#
warnings.filterwarnings("ignore")
# equivalents from matplotlib
# data = pd.read_csv("dfe.csv",encoding = "ISO-8859-1")
# print(data[data.columns[1]])
data = pd.read_csv('neurotissues.csv')
print(data.iloc[21])
# print("21,2=",data.values[21, 2])
# print(data)
# print(data.columns.values)
x = data.values
for j in range(1,691):
    if x[21,j] == 'N':
        x[21,j] = 1
    if x[21,j] == 'A':
        x[21,j] = 0
    if x[31,j] == 'N':
        x[31, j] = 1
    if x[31,j] == 'A':
        x[31, j] = 0
# print(x[21,:])
# print(x[31,:])

arrayY = x[21,1:691]
arrayX = x[56:39335, 1:]

arrayX = np.transpose(arrayX)
arrayY = np.transpose(arrayY)
print("arrX=", arrayX.shape)
print("arrY=", arrayY.shape)
# dont use 31 there are alot of NA values
# np.save('arrays/X1', X1,allow_pickle=True)
# np.save('arrays/Y1', Y1,allow_pickle=True)
# arrayX = np.load('arrays/X1.npy', allow_pickle=True)
# arrayY = np.load('arrays/Y1.npy',allow_pickle=True)
#
# arrayX = arrayX[1:, :]
# arrayY = arrayY[1:, :]
# # arrayY = np.load('arrays/only_pg.npy', allow_pickle=True)
# # arrayY = np.transpose(arrayY)
# # Topic = 3
# # arrayY[arrayY!=Topic]=-1
# # arrayY[arrayY==Topic] = 1
# # arrayY[arrayY==-1] = 0
#
# # index = np.where(arrayX[:,2]==0)
# # arrayX = arrayX[index]
# # arrayY = arrayY[index]
# np.reshape(arrayY,(690, 1))
# print(arrayX, arrayY)
# print(arrayX.shape, arrayY.shape)
#
#
# # [chi2,pval] = sk.chi2(arrayX, arrayY)
# # print("chi2 value for gender, state, followers with category", chi2)
# # print("these are the pvalues=", pval)

z = np.concatenate((arrayX, arrayY[:,None]), axis = 1)
print("adjoined=",z)
np.random.shuffle(z)
print("shuffled=",z)

print("80 percent is at", int(0.8*len(z)))
Xtrain = z[:int((.8*len(z))),:9]
Ytrain = z[:int((.8*len(z))),9]
Xtest = z[int(0.8*len(z)):,:9]
Ytest = z[int(0.8*len(z)):,9]
Ytrain = Ytrain.astype(float)
Xtrain = Xtrain.astype(float)
Ytest = Ytest.astype(float)
Xtest = Xtest.astype(float)
Ytrain = np.around(Ytrain, decimals =3) #this is to remove floats
Ytrain = Ytrain * 1000 # this is to remove floats
print("Xtrain = ", Xtrain)
print("Ytrain = ", Ytrain)
Ytrain.astype(int) #this is to change type to int for the bincount

# SMOTE ###################################################
print("before =", np.bincount(Ytrain))
print(Xtrain.shape,Ytrain.shape)
smt = SMOTE()
Xtrain, Ytrain = smt.fit_sample(Xtrain, Ytrain)
print("after =", np.bincount(Ytrain))
print(Xtrain.shape,Ytrain.shape)
print(Xtrain.shape,Ytrain.shape)

print("Percentage of each")
print(Ytest.shape)
count = np.bincount(Ytest)
print(count)
print(count[0]/Ytest.shape[0], count[1]/Ytest.shape[0])

# decision tree ############################################3
tr = DecisionTreeClassifier()
tr.fit(Xtrain, Ytrain)
# print("\nimportances=",tr.feature_importances_)


y_predict = tr.predict(Xtest)
print(f"Accuracy score for Decision Tree Classifier is: {accuracy_score(Ytest, y_predict)}")
print("DT importances=",tr.feature_importances_)

cm = ConfusionMatrix(tr, classes=[0,1])
cm.fit(Xtrain, Ytrain)
cm.score(Xtest, Ytest)
cm.show()

#Logistic Regression#####################################

tr2 = LogisticRegression()
tr2.fit(Xtrain, Ytrain)
# print("importances =", tr2.feature_importances_)


y_predict = tr2.predict(Xtest)
print(f"\nAccuracy score for Logistic Regression Classifier is: {accuracy_score(Ytest, y_predict)}")
print("RL importances=",tr2.coef_)
# print("Ypred=", y_predict)
#confusion ConfusionMatrix only commented bc lab doesnt have yellow brick
cm = ConfusionMatrix(tr2, classes=[0,1])
cm.fit(Xtrain, Ytrain)
cm.score(Xtest, Ytest)
cm.show()


# add Random FOrest Classifier #############################
rf = RandomForestClassifier()
rf.fit(Xtrain, Ytrain)
# print("\nimportances =", rf.feature_importances_)

y_predict = rf.predict(Xtest)
print(f"\nAccuracy score for Random Forest Classifier is: {accuracy_score(Ytest, y_predict)}")
print("RF importances=",rf.feature_importances_)

cm = ConfusionMatrix(rf, classes=[0,1])
cm.fit(Xtrain, Ytrain)
cm.score(Xtest, Ytest)
cm.show()

# Gaussian Process Classifier ######################3############
# gp = GaussianProcessClassifier()
# gp.fit(Xtrain, Ytrain)
# # print("importances=", gp.feature_importances_)
#
# y_predict = gp.predict(Xtest)
# print(f"Accuracy score for Gaussian Process Classifier Classifier is: {accuracy_score(Ytest, y_predict)}")
# # print("GP importances=", gp.feature_importances_)
#
# cm = ConfusionMatrix(gp, classes=[0,1])
# cm.fit(Xtrain, Ytrain)
# cm.score(Xtest, Ytest)
# cm.show()

# MLP Classifier #############################################
mlp = MLPClassifier()
mlp.fit(Xtrain, Ytrain)
# print("importances=", gp.feature_importances_)

y_predict = mlp.predict(Xtest)
print(f"\nAccuracy score for MLP Classifier Classifier is: {accuracy_score(Ytest, y_predict)}")
# print("GP importances=", gp.feature_importances_)

cm = ConfusionMatrix(mlp, classes=[0,1])
cm.fit(Xtrain, Ytrain)
cm.score(Xtest, Ytest)
cm.show()

# AdaBoostClassifier #######################################
ada = AdaBoostClassifier()
ada.fit(Xtrain, Ytrain)
# print("\nimportances=", ada.feature_importances_)

y_predict = ada.predict(Xtest)
print(f"\nAccuracy score for Ada Boost Classifier is: {accuracy_score(Ytest, y_predict)}")
print("ADA importances=", ada.feature_importances_)

cm = ConfusionMatrix(ada, classes=[0,1])
cm.fit(Xtrain, Ytrain)
cm.score(Xtest, Ytest)
cm.show()

# Quadratic Discriminant Analysis #######################################
qda = QuadraticDiscriminantAnalysis()
qda.fit(Xtrain, Ytrain)
# print("importances=", qda.feature_importances_)

y_predict = qda.predict(Xtest)
print(f"\nAccuracy score for Quad Discriminant Analysis Classifier is: {accuracy_score(Ytest, y_predict)}")
# print("ADA importances=", qda.feature_importances_)

cm = ConfusionMatrix(qda, classes=[0,1])
cm.fit(Xtrain, Ytrain)
cm.score(Xtest, Ytest)
cm.show()

# Gaussian NB #######################################
gnb = GaussianNB()
gnb.fit(Xtrain, Ytrain)
# print("importances=", qda.feature_importances_)

y_predict = gnb.predict(Xtest)
print(f"\nAccuracy score for GaussianNB Classifier is: {accuracy_score(Ytest, y_predict)}")
# print("ADA importances=", qda.feature_importances_)

cm = ConfusionMatrix(gnb, classes=[0,1])
cm.fit(Xtrain, Ytrain)
cm.score(Xtest, Ytest)
cm.show()

# SVC Classifier #######################################################
svc = SVC()
svc.fit(Xtrain, Ytrain)
# print("importances=", qda.feature_importances_)

y_predict = svc.predict(Xtest)
print(f"\nAccuracy score for SVC Classifier is: {accuracy_score(Ytest, y_predict)}")
# print("ADA importances=", qda.feature_importances_)

cm = ConfusionMatrix(svc, classes=[0,1])
cm.fit(Xtrain, Ytrain)
cm.score(Xtest, Ytest)
cm.show()

# Kneighbors Classifier #######################################################
kn = KNeighborsClassifier()
kn.fit(Xtrain, Ytrain)
# print("importances=", qda.feature_importances_)

y_predict = kn.predict(Xtest)
print(f"\nAccuracy score for KNeighbors Classifier is: {accuracy_score(Ytest, y_predict)}")
# print("ADA importances=", qda.feature_importances_)

cm = ConfusionMatrix(kn, classes=[0,1])
cm.fit(Xtrain, Ytrain)
cm.score(Xtest, Ytest)
cm.show()
