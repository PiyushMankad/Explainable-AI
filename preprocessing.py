import os
import glob
import re
import readability

from NLP_strategy import normalize_corpus

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def readable(file):
	with open(file,"r+") as fp:
		kicker = fp.readline()
		filecontent = fp.readlines()
		readability_score = readability.getmeasures(filecontent,lang="en")

	norm_corpus, filecontent, clean_text = normalize_corpus(filecontent)
	# print('\n\n Normalized Corpus\n {} \n \n Kicker \n {} \nReadablility score\n {}'.format(norm_corpus,kicker,readability_score))
	print("Norm corpus\n {} \n\n File content\n {} \n\n Clean text\n {}".format((norm_corpus),(filecontent),(clean_text)))



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


if __name__ == '__main__':
	traverseArticles()