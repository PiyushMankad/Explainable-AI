import os
import glob
import re
import readability


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def readable(file):
	with open(file,"r+") as fp:
		filecontent = fp.readlines()

	print(filecontent)



def traverseArticles():
	path = os.getcwd()
	print(path)
	counter = 1
	for files in sorted(glob.glob(path+"\\*.txt"),key=numericalSort):
		# print(files)
		readable(files)
		if counter > 1:
			break
		counter+=1


if __name__ == '__main__':
	traverseArticles()