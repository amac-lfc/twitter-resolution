#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries and Data

# In[1]:


import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import numpy as np
import matplotlib.pyplot as plt


# data = pd.read_csv("SMSSpamCollection.tsv", names=['gender', 'text'], sep='\t')
data = pd.read_excel('dfe2.xlsx',  usecols = ['Personal Growth', 'text'])
print(data.head())


# In[2]:


data['Personal Growth'].value_counts()


# ### Preprocessing Data

# In[3]:


stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer() #

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

data['body_len'] = data['text'].apply(lambda x: len(x) - x.count(" "))
data['punct%'] = data['text'].apply(lambda x: count_punct(x))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text


# ### Split into train/test

# In[4]:


from sklearn.model_selection import train_test_split

X=data[['text', 'body_len', 'punct%']]
y=data['Personal Growth']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# ### Vectorize text

# In[5]:


tfidf_vect = TfidfVectorizer(analyzer=clean_text)
tfidf_vect_fit = tfidf_vect.fit(X_train['text'])

tfidf_train = tfidf_vect_fit.transform(X_train['text'])
tfidf_test = tfidf_vect_fit.transform(X_test['text'])

X_train_vect = pd.concat([X_train[['body_len', 'punct%']].reset_index(drop=True),
           pd.DataFrame(tfidf_train.toarray())], axis=1)
X_test_vect = pd.concat([X_test[['body_len', 'punct%']].reset_index(drop=True),
           pd.DataFrame(tfidf_test.toarray())], axis=1)

X_train_vect.head()
print("ytrain", y_train, "\n\n\n")

# ### Final evaluation of models
bins = np.linspace(0, 200, 40)

plt.hist(data[data['Personal Growth']=='1']['body_len'] , bins, alpha = 0.5, normed = True, label = 'male')
plt.hist(data[data['Personal Growth']=='0']['body_len'], bins, alpha = 0.5, normed = True, label = 'female')
plt.legend(loc = 'upper left')
plt.show()

bins = np.linspace(0, 50, 40)

plt.hist(data[data['Personal Growth']=='1']['punct%'],  bins, alpha = 0.5, normed = True, label = 'male')
plt.hist(data[data['Personal Growth']=='0']['punct%'],  bins, alpha = 0.5, normed = True, label = 'female')
plt.legend(loc = 'upper left')
plt.show()
# In[6]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as acs
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)

rf_model = rf.fit(X_train_vect, y_train)

y_pred = rf_model.predict(X_test_vect)

precision, recall, fscore, train_support = score(y_test, y_pred, pos_label='male', average='binary')
print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(y_test,y_pred), 3)))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
class_label = ["1- yes", "0 - no"]
df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
sns.heatmap(df_cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
