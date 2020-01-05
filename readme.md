# Independent Project
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

## 1.a) Twitter Resolution Histogram
#### Code can be found in histogram.py
#### Initially, you may want to look at the data in various ways. In this first plot we look at Tweet Frequency by State and Gender, which gives us an idea of which states tweet the most, and whether one gender or the other is more dominant on Twitter.
![alt text](https://github.com/lfc-math-cs/twitter-resolution/blob/master/TweetFreqByStateGender.png)

#### In the next plot we look at the Frequency of Tweets in each state color coding for Category. This shows us in general, the topic of each tweet.
![alt text](https://github.com/lfc-math-cs/twitter-resolution/blob/master/TweetFreqByStateCategory.png)

#### To take an even closer look, this plot shows us the number of tweets in each state that belong to a certain category. For this example, the category chosen was Personal Growth.

![alt text](https://github.com/lfc-math-cs/twitter-resolution/blob/master/SingleCategoryByState.png)

## Classifiers

## Alzheimer's dataset
