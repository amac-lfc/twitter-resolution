# Part 1: The Analysis of New Year's Resolutions on Twitter

#### This project is conducted using data from data world. The goal of this project is to use Machine Learning on a dataset containing Tweets of New Years Resolutions and to use various characteristics such as State, Gender, Followers, and Language Quantifiers in order to predict the Category of the Tweet.

## Outline
#### 1.Twitter Resolution Data
####  a. Histogram
####  b. Chi Square test
####  c. Classifiers
#### 2. Alzheimer's Dataset
####  a. Classifiers
## Install all necessary packages
~~~~~~~~~~~~~{.python}
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
~~~~~~~~~~~~~~

## 1a) Twitter Resolution Histogram
#### Code can be found in histogram.py
#### Initially, you may want to look at the data in various ways. In this first plot we look at Tweet Frequency by State and Gender, which gives us an idea of which states tweet the most, and whether one gender or the other is more dominant on Twitter.
![alt text](https://github.com/lfc-math-cs/twitter-resolution/blob/master/TweetFreqByStateGender.png)

#### In the next plot we look at the Frequency of Tweets in each state color coding for Category. This shows us in general, the topic of each tweet.
![alt text](https://github.com/lfc-math-cs/twitter-resolution/blob/master/TweetFreqByStateCategory.png)

#### To take an even closer look, this plot shows us the number of tweets in each state that belong to a certain category. For this example, the category chosen was Personal Growth.

![alt text](https://github.com/lfc-math-cs/twitter-resolution/blob/master/SingleCategoryByState.png)

## 1c) Classifiers
#### Code can be found in classifier.py
#### In this portion I will run several classifiers to see which best predicts the tweet category. The classifiers used are SMOTE, Decision Tree, Logistic Regression, Random Forest Classifier, MLP Classifier, AdaBoost Classifier, Quadratic Discriminant Analysis, Gaussian NB, SVC Classifier, and KNeighbors Classifier.

#### For each classifier, I trained and fit the data, and then I created the Confusion matrix. Decision Tree Classifier shown as an example.

~~~~{.python}
tr = DecisionTreeClassifier()
tr.fit(Xtrain, Ytrain)
y_predict = tr.predict(Xtest)
print(f"Accuracy score for Decision Tree Classifier is: {accuracy_score(Ytest, y_predict)}")
print("DT importances=",tr.feature_importances_)

cm = ConfusionMatrix(tr, classes=[0,1])
cm.fit(Xtrain, Ytrain)
cm.score(Xtest, Ytest)
cm.show()
~~~~

## Decision Tree Classifier

![alt text](https://github.com/lfc-math-cs/twitter-resolution/blob/master/DecisionTreeCfsnMatrix.png)

#### From this confusion matrix, we can see the number of times that the classifier guessed correct. In this case, 636/985 is the fraction of times that the classifier guessed the actual value, which means it is certainly better than random guessing.

# Part 2: The Analysis of Patients and the Diagnosis of Alzheimer's disease
#### Code can be found in geoclassifiers.py
