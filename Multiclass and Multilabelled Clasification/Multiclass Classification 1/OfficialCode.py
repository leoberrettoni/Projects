import pandas as pd
from sklearn.model_selection import train_test_split
import gensim

import numpy as np
from numpy import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re

# print working directory
import os
print(os.getcwd())
# set working directory
os.chdir('C:\\Users\\vinci\\OneDrive\\Desktop\\day1')
df = pd.read_csv('train_set.csv')

df['text_clean'] = df['Text'].apply(lambda x: gensim.utils.simple_preprocess(x))
df.head()


X = df['text_clean']
y = df['Directory code']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state= 42)


w2v_model = gensim.models.Word2Vec(X_train,
                                   vector_size=100,
                                   window=5,
                                   min_count=2)

words = set(w2v_model.wv.index_to_key )

X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_train])

X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_test])


for i, v in enumerate(X_train_vect):
    print(len(X_train.iloc[i]), len(v))
    



X_train_vect_avg = []
for v in X_train_vect:
    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))
    else:
        X_train_vect_avg.append(np.zeros(100, dtype=float))
        
X_test_vect_avg = []
for v in X_test_vect:
    if v.size:
        X_test_vect_avg.append(v.mean(axis=0))
    else:
        X_test_vect_avg.append(np.zeros(100, dtype=float))
        
        
        
        
        
for i, v in enumerate(X_train_vect_avg):
   print(len(X_train.iloc[i]), len(v))      
        
        
        
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_model = rf.fit(X_train_vect_avg, y_train.values.ravel())
        
        
y_pred = rf_model.predict(X_test_vect_avg)
        
        
# y_pred to csv
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('y_pred.csv', index=False)
