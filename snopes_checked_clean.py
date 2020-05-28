import pandas as pd
from text_features import get_noun_phrase,get_text_features

# file = str("snopes_checked_v02.csv")
data = pd.read_csv("snopes_checked.csv")
labels = list(data['article_category_phase1'])
title = data['article_title_phase1']
text = data['original_article_text_phase2']

print("Starting")
texts = []
for i,j in zip(title,text):
	texts.append(i + " " + j)

	



noun,properNoun = get_text_features(texts)
print("Nouns and Propn extracted")

noun_phrases = pd.DataFrame(get_noun_phrase(texts),columns=["Noun Phrases"])
noun_phrases["Labels"] = labels
for i in range(len(labels)):
	if "Politics" in str(labels[i]):
		labels[i] = "Politics"
	elif "Science" in str(labels[i]):
		labels[i] = "Health & Science"
	elif "Medical" in str(labels[i]):
		labels[i] = "Health & Science"
	else:
		continue
print(labels)
noun_phrases["New_Labels"] = labels
noun_phrases["Noun"] = noun
noun_phrases["Propernoun"] = properNoun
noun_phrases.to_csv("snopesBuzzfeed_textFeatures2.csv",index=False)