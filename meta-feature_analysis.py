# -*- coding: utf-8 -*-
"""
Created on Thu Mar 5 11:10:50 2020

@author: mankadp
"""
import os,re
import pandas as pd
import readability
import numpy as np
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer



## gets LENGTH FEATURES from the text



def counting_features(texts):
    pronounCount = np.zeros(len(texts)).reshape(-1,1)
    prepositionCount = np.zeros(len(texts)).reshape(-1,1)
    conjunctionCount = np.zeros(len(texts)).reshape(-1,1)

    complexWordsCount = np.zeros(len(texts)).reshape(-1,1)
    longWordsCount = np.zeros(len(texts)).reshape(-1,1)
    syllablesCount = np.zeros(len(texts)).reshape(-1,1)

    ## type token ratio is the no of unique words divided by total words
    typeTokenRatio = np.zeros(len(texts)).reshape(-1,1)
    wordCount = np.zeros(len(texts)).reshape(-1,1)


    for i,text in enumerate(texts):
        # print(text)
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

    ## Combining all of them into one
    featureCounts = pd.DataFrame(data = np.concatenate((pronounCount,prepositionCount,conjunctionCount,complexWordsCount
                                           ,longWordsCount,syllablesCount,typeTokenRatio,wordCount),axis=1),
    columns=["pronounCount","prepositionCount","conjunctionCount","complexWordsCount","longWordsCount",
             "syllablesCount","typeTokenRatio","wordCount"])
    return featureCounts


def preprocessing(texts):
    nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english')
    cleanedText = []
    entity = []

    for i,text in enumerate(texts):
        words = ""
        temp_entity = ""
        ## Lemmatization
        text = nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])

        ## removes Special Characters
        pattern = r'[^a-zA-z0-9\s]'
        text = re.sub(pattern,"",str(text))
        # print(text)

        ## seperates each word and lowercases as well
        tokens = tokenizer.tokenize(text.lower())
        ## removes spaces in each words, if any
        tokens = [token.strip() for token in tokens]
        ## filters stop words in tokens
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        for word in filtered_tokens:
            words += word + " "
        cleanedText.append(words)

        ## Named entity recognition
        words = nlp(words)
        temp_entity += str([word for word in words if word.ent_type])
        entity.append(temp_entity)

        # print(words)
        # print(cleanedText)
        # break
        if i%500 == 0:
            print("Steps Done:",i)

    return cleanedText,entity


if __name__ == "__main__":

    STEPS = 2000
    data = pd.read_csv("train.csv").iloc[:STEPS,:]

    try:
        os.system("pip install -r requirements.txt")
        nltk.download('stopwords')
        os.system("python -m spacy download en_core_web_sm")
        os.system("python -m textblob.download_corpora")
    except:
        pass
    '''
    '''

    feature = data.iloc[:,1]
    texts = data['text']

# =============================================================================
    # Consists of all the counted features
    featureCounts=counting_features(texts)
    print("Feature Counts DONE!")
# =============================================================================


# =============================================================================
    # Lemmatization, Removing special characters, Stop words Removal
    cleanedText = preprocessing(texts)
    print("Text Cleaning DONE!")
# =============================================================================
