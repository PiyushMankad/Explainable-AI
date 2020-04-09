# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 00:32:07 2020

@author: Piyush

Decision Trees

"""


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv(r"D:\Intelligent Systems\Dissertation ####\Explainable-AI\subdata.csv").iloc[:10000,:4]
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
#print(oe.categories_)

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
print("tfidf1")

"""
tfidf2 = TfidfVectorizer()
cleanText = pd.DataFrame(tfidf2.fit_transform(cleanText).toarray())
cleanTextTest = pd.DataFrame(tfidf2.transform(cleanTextTest).toarray())
print("tfidf 2")

### joining all again
X_train = pd.concat([(X_train.iloc[:,0]),normal,cleanText],axis=1)
X_test = pd.concat([(X_test.iloc[:,0]),normalTest,cleanTextTest],axis=1)
"""

### Training a model
from sklearn import tree
#clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=42)
clf = clf.fit(normal,Y_train)
print("model fitted")

Y_predict = clf.predict(normalTest)
Y_predict_probability = clf.predict_proba(normalTest)
tree.plot_tree(clf)
print("predicted")

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("D:\Intelligent Systems\Dissertation ####\Explainable-AI\DT10-multiclass")

# Evaluation metrics
from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,precision_recall_fscore_support
confMatrix = confusion_matrix(Y_test,Y_predict)
accuracy = accuracy_score(Y_test,Y_predict)
precision_recall_fscore = precision_recall_fscore_support(Y_test,Y_predict)

score = np.sqrt(mean_squared_error(Y_test,Y_predict))
print("\n Final Accuracy Score is",accuracy)

