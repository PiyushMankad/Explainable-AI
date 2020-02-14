# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:10:50 2020

@author: mankadp
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv(r"E:\Intelligent Systems\Dissertation ####\Explainable-AI\subdata.csv").iloc[:1000,:4]
data = data.fillna("undefined")
labels = data.iloc[:,0]
print(type(labels))
X = data.iloc[:,1:]
print("imported")

#### transforming labels
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
labels = oe.fit_transform(pd.DataFrame(labels).to_numpy().reshape(-1,1))

print("ordinal Encoder")
print(oe.categories_)

### categorisation
for i,label in enumerate(labels):
    if label == 36:
        labels[i] = 1
    else:
        labels[i] = 0
labels = pd.DataFrame(labels)

### train trst split
X_train,X_test,Y_train,Y_test = train_test_split(X.iloc[:,:], labels, test_size=0.1,shuffle = False)
normal = X_train.iloc[:,1]
cleanText = X_train.iloc[:,2]
normalTest = X_test.iloc[:,1]
cleanTextTest = X_test.iloc[:,2]
print("split")

### TFIDF Vector

tfidf = TfidfVectorizer()
normal = pd.DataFrame(tfidf.fit_transform(normal).toarray())
normalTest = pd.DataFrame(tfidf.transform(normalTest).toarray())

    	
tfidf2 = TfidfVectorizer()
cleanText = pd.DataFrame(tfidf2.fit_transform(cleanText).toarray())
cleanTextTest = pd.DataFrame(tfidf2.transform(cleanTextTest).toarray())
print("tfidf 2")

### joining all again
X_train = pd.concat([(X_train.iloc[:,0]),normal,cleanText],axis=1) 
X_test = pd.concat([(X_test.iloc[:,0]),normalTest,cleanTextTest],axis=1)


### Training a model
import lightgbm
train_data = lightgbm.Dataset(normal, label=Y_train)
test_data = lightgbm.Dataset(normalTest, label=Y_test)


parameters = {
    'objective': 'binary',
    'max_depth':30,
    'learning_rate': 0.1,
    'metric': 'mae',
    'feature_fraction': 0.8,
    'boosting': 'gbdt',
    'bagging_freq': 20,
    'verbose': -1
}

params = {
        'task': 'train',
        'obective': 'binary',
        'metric':'binary_error',
        'verbose':1
        }
model = lightgbm.train(params,
                       train_data,
                       valid_sets=[train_data,test_data],
                       verbose_eval=10,
                       num_boost_round=1000,
                       early_stopping_rounds=100)

Y_predict = np.round(model.predict(normalTest))

from sklearn.metrics import mean_squared_error,confusion_matrix
confMatrix = confusion_matrix(Y_test,Y_predict)

score = np.sqrt(mean_squared_error(Y_test,Y_predict))

print("\n Final Score is",score)

### saving specific variables
import time


	