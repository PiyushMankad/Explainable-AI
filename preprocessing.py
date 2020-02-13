import os
import glob
import re
import readability
from NLP_strategy import normalize_corpus
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import pandas as pd



numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def generateCSV(kicker,readability_score,norm_corpus,clean_text):
	with open("data.csv","a+",newline='') as file:
		writer = csv.writer(file)
		writer.writerow([kicker,readability_score,norm_corpus,clean_text])
	print(kicker,readability_score,norm_corpus,clean_text)


def convertNormCorpus(norm_corpus):
	words = ""
	norm_corpus = norm_corpus.replace("]",",").replace("["," ")
	# print(norm_corpus)
	for word in norm_corpus.split(","):
		words += word

	return words


def readable(file):
	with open(file,"r+") as fp:
		kicker = fp.readline()
		filecontent = fp.readlines()

	norm_corpus, filecontent, clean_text = normalize_corpus(filecontent)
	readability_score = readability.getmeasures(filecontent,lang="en")
	norm_corpus=convertNormCorpus(norm_corpus)
	# print('\n\n Normalized Corpus\n {} \n \n Kicker \n {} \nReadablility score\n {}'.format(norm_corpus,kicker,readability_score))
	# print("Norm corpus\n {} \n\n File content\n {} \n\n Clean text\n {}".format(type(norm_corpus),type(filecontent),type(clean_text)))

	generateCSV(kicker,readability_score['readability grades']['FleschReadingEase'],norm_corpus,clean_text)

def traverseArticles():
	path = os.getcwd()
	print(path)
	counter = 1
	for files in sorted(glob.glob(path+"\\*.txt"),key=numericalSort):
		# print(files)
		if counter > 1:
			break
		readable(files)
		counter+=1



def make_vectors(doc):
	vector = TfidfVectorizer()
	docMatrix = vector.fit_transform(doc)
	print(vector.get_feature_names())
	print(docMatrix.shape)
  	



def preprocessing():
	data = pd.read_csv(r"E:\Intelligent Systems\Dissertation ####\Explainable-AI\data.csv")
	labels = data.iloc[:,0]
	X = data.iloc[:,1:-1]


if __name__ == '__main__':
	traverseArticles()
	# preprocessing()