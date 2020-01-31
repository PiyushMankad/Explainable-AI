import os
import glob
import re

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def traverseArticles():
	path = os.getcwd()
	print(path)
	for files in sorted(glob.glob(path+"\\*.txt"),key=numericalSort):
		print(files)


if __name__ == '__main__':
	traverseArticles()