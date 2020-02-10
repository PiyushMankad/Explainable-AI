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

	norm_corpus = normalize_corpus(filecontent)
	# print(filecontent,kicker)
	print(norm_corpus,kicker)



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