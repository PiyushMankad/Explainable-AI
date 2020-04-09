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



def counting_features(texts):
    pronounCount = np.zeros(len(texts)).reshape(-1,1)
    prepositionCount = np.zeros(len(texts)).reshape(-1,1)
    conjunctionCount = np.zeros(len(texts)).reshape(-1,1)

    complexWordsCount = np.zeros(len(texts)).reshape(-1,1)
    longWordsCount = np.zeros(len(texts)).reshape(-1,1)
    syllablesCount = np.zeros(len(texts)).reshape(-1,1)

    # type token ratio is the no of unique words divided by total words
    typeTokenRatio = np.zeros(len(texts)).reshape(-1,1)
    wordCount = np.zeros(len(texts)).reshape(-1,1)


    for i,text in enumerate(texts):
#        print(text)
        score = readability.getmeasures(text,lang="en")
        sentenceInfo = score["sentence info"]
        wordUsage = score["word usage"]
        # word usages
        pronounCount[i] = wordUsage['pronoun']
        prepositionCount[i] = wordUsage['preposition']
        conjunctionCount[i] = wordUsage['conjunction']
        # sentence info
        complexWordsCount[i] = sentenceInfo['complex_words']
        longWordsCount[i] = sentenceInfo['long_words']
        syllablesCount[i] = sentenceInfo['syllables']
        typeTokenRatio[i] = sentenceInfo['type_token_ratio']
        wordCount[i] = sentenceInfo['words']

    # Combining all of them into one
    featureCounts = pd.DataFrame(data = np.concatenate((pronounCount,prepositionCount,conjunctionCount,complexWordsCount
                                           ,longWordsCount,syllablesCount,typeTokenRatio,wordCount),axis=1),
    columns=["pronounCount","prepositionCount","conjunctionCount","complexWordsCount","longWordsCount",
             "syllablesCount","typeTokenRatio","wordCount"])
    return featureCounts

if __name__=="__main__":

    data = pd.read_csv(r"D:\Intelligent Systems\Dissertation ####\Explainable-AI\subdata.csv").iloc[:10000,:4]
    data = data.fillna("undefined")
    labels_init = data.iloc[:,0]

    print(type(labels_init))

    X = data.iloc[:,1:]
    print("imported")

    #### transforming labels
    from sklearn.preprocessing import OrdinalEncoder
    oe = OrdinalEncoder()
    labels = oe.fit_transform(pd.DataFrame(labels_init).to_numpy().reshape(-1,1))

    print("ordinal Encoder")
    print(oe.categories_)

    ### categorisation
    '''
    for i,label in enumerate(labels):
        if label == 36:
            labels[i] = 1
        if label == 31:
            labels[i] = 2
        else:
            labels[i] = 0
    labels = pd.DataFrame(labels)
    '''

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
