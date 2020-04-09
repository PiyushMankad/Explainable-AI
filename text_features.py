import spacy
import pandas as pd
import pytextrank

def get_text_features(file):
    pass
data = pd.read_csv("subdata.csv").fillna("undefined")
text = data["Clean"]

## text features
noun = []

## text seperation
nlp = spacy.load("en_core_web_sm")
# add PyTextRank to the spaCy pipeline
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

for sentence in text:
    continue
    doc = nlp(sentence)
    # print(sentence)
    ## examine the top-ranked phrases in the document
    for p in doc._.phrases:
        print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
        print(p.chunks)

    addNoun = ""
    for token in doc:
        # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              # token.shape_, token.is_alpha, token.is_stop)
        if token.pos_ == "NOUN":
            # print("its a noun",token.text)
            addNoun += token.text + " "

    noun.append(addNoun)
    break





