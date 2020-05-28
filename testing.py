# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:56:17 2020

@author: Piyush
"""

import glob
import os
from text_features import get_text_features, get_noun_phrase
from NLP_strategy import normalize_corpus
import pandas as pd

def readArticles(path):
	lines = []
	label = []
	source = []
	for file in os.listdir(path):
		with open(path+"\\"+file,"r",encoding="utf8") as fp:
# 		print(lines)
			normalize_corpusnorm_corpus, filecontent, clean_text = normalize_corpus(fp.readlines())
		lines.append(clean_text)
		label.append(file.split("_")[1])
		source.append(file.split("_")[0])

	return lines,label,source

if __name__ == "__main__":
	path = r"D:\Intelligent Systems\Dissertation ####\testing"
	texts,label,source = readArticles(r"D:\Intelligent Systems\Dissertation ####\testing")
	noun,properNoun = get_text_features(texts)
	nounPhrase = get_noun_phrase(texts)
	data = pd.DataFrame(label,columns=["label"])
	data["source"] = source
	data["noun"] = noun
	data["file"] = os.listdir(path)
	data["properNoun"] = properNoun
	data["nounPhrase"] = nounPhrase
	data.to_csv("test_newsources.csv",index=False)
