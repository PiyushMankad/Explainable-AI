import spacy
import pandas as pd
import pytextrank
import re, os
import analytics
from textblob import TextBlob


def get_text_features(texts):
    noun = []
    properNoun = []
    rank = {}

    ## text seperation
    nlp = spacy.load("en_core_web_sm")
    # add PyTextRank to the spaCy pipeline
    tr = pytextrank.TextRank()
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

    count = 0
    for sentence in texts:
        doc = nlp(sentence)

        ## examine the top-ranked phrases in the document
        for p in doc._.phrases:
            rank[p.text] = p.rank

        ## parts of speech tagging
        addNoun = ""
        addPnoun = ""
        for token in doc:
            # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                  # token.shape_, token.is_alpha, token.is_stop)
            if token.pos_ == "NOUN":
                # print("its a noun",token.text)
                addNoun += token.text.replace("_"," ") + " "
            if token.pos_ == "PROPN":
                # print("its a proper noun",token.text)
                addPnoun += token.text.replace("_"," ") + " "

        noun.append(addNoun)
        properNoun.append(addPnoun)
        count += 1
        if count%100 == 0:
            print(count,"Iterations are DONE")
    return noun,properNoun

def get_noun_phrase(texts):
    noun_phrase = []
    count = 0
    for text in texts:
        word  = ""
        blob = TextBlob(text)
        phrases = blob.noun_phrases
        for phrase in phrases:
            phrase = phrase.replace(" ","_")
            phrase = phrase.replace(".","")
            word += phrase + " "
        # print(word)
        noun_phrase.append(word)
        count += 1
        if count%100 == 0:
            print(count,"Iterations are DONE")

    return noun_phrase


if __name__ == "__main__":

    STEPS = 5000

    data = pd.read_csv("subdata.csv").iloc[:STEPS,:].fillna("undefined")
    texts = data["Clean"]
    labels = data["Labels"]

    ## text features
    noun,properNoun = get_text_features(texts)
    print("Nouns and Propn extracted")

    ## noun phrases
    noun_phrases = pd.DataFrame(get_noun_phrase(texts),columns=["Noun Phrases"])
    noun_phrases["Labels"] = labels
    noun_phrases["Noun"] = noun
    noun_phrases["Propernoun"] = properNoun
    noun_phrases.to_csv("all_nouns5000.csv",index=False)


    ## extracting counted features
    # featureCounts = analytics.counting_features(texts)

    # ## making a CSV
    # featureCounts['Noun'] = noun
    # featureCounts['Labels'] = labels
    # featureCounts.to_csv("proper_noun_train.csv",index=False)

