# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:29:56 2020

@author: Piyush
"""


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import readability

# =============================================================================
# First run text_features.py to make an appropriate csv
# =============================================================================

# data = pd.read_csv("proper_noun_train.csv").fillna("undefined")
data = pd.read_csv("noun_phrases_larger.csv").fillna("undefined")
labels_init = data["Labels"]

print(type(labels_init))

X = data.iloc[:,:9]
X = data.iloc[:,0]
print("imported")

#### transforming labels
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
labels = oe.fit_transform(pd.DataFrame(labels_init).to_numpy().reshape(-1,1))

print("ordinal Encoder")
print(oe.categories_)

### TFIDF Vector
tfidf = TfidfVectorizer()
noun = np.array(tfidf.fit_transform(X).toarray(),dtype = np.int32)

noun = tfidf.fit_transform(X)
print(type(noun))
print("tfidf1")

## joining
# X = pd.concat([X.iloc[:,:8],noun],axis = 1)

### train trst split
X_train,X_test,Y_train,Y_test = train_test_split(noun.data, labels, test_size=0.1,shuffle = True)
# noun = X_train["Noun"]
# nounTest = X_test["Noun"]


# print(tfidf.get_feature_names())


## joining
# X_train = pd.concat([(X_train.iloc[:,:-2]),noun],axis=1)
# X_test = pd.concat([(X_test.iloc[:,:-2]),nounTest],axis=1)


### Training a model
from sklearn.model_selection import GridSearchCV
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=42)
clf.fit(X_train,Y_train)
# clf.fit(noun,Y_train)
print("model fitted")

##Prediction
Y_predict = clf.predict(X_test)
# Y_predict = clf.predict(nounTest)
Y_predict_probability = clf.predict_proba(X_test)
	print("predicted")

import time
t1 = time.time()
## cross validation (experiment 3.2)
treeclf = tree.DecisionTreeClassifier()
parameters = {'criterion':['gini'],'max_depth':[100,150],
              'min_samples_split':[6,8,10],'min_samples_leaf':[1],
              'max_features':["sqrt"],"min_weight_fraction_leaf":[0],
              'random_state':[42]}

n_jobs = -1 # utilises all the processors
clf2  = GridSearchCV(treeclf,parameters,n_jobs=-1)
clf2.fit(X_train,Y_train)
# clf2.fit(noun,Y_train)
print("cross validation ran")
print("Time taken for cross validation to run: ",time.time()-t1)

## getting the best metric results
cv_results = clf2.cv_results_
best_estimator = clf2.best_estimator_
best_params = clf2.best_params_


## prediction for cross validaion
Y_predictcv = clf2.predict(X_test)
# Y_predictcv = clf2.predict(nounTest)
Y_predict_probabilitycv = clf2.predict_proba(X_test)
print("predicted CV")


# import graphviz
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("D:\Intelligent Systems\Dissertation ####\Explainable-AI\DT10-multiclass")


## Evaluation metrics
from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,precision_recall_fscore_support
confMatrix = confusion_matrix(Y_test,Y_predict)
accuracy_nounph = accuracy_score(Y_test,Y_predict)
precision_recall_fscore = precision_recall_fscore_support(Y_test,Y_predict)

## Evaluation metric for cross validation
confMatrix_cv = confusion_matrix(Y_test,Y_predictcv)
accuracy_cv_noun_phrase = accuracy_score(Y_test,Y_predictcv)
precision_recall_fscore_cv = precision_recall_fscore_support(Y_test,Y_predict)

score = np.sqrt(mean_squared_error(Y_test,Y_predict))
print("\n Final Accuracy Score is",accuracy_nounph)
print("\n Final accuracy for Grid search CV is:",accuracy_cv_noun_phrase)



