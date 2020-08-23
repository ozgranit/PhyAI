import csv
import os
import pandas as pd
import numpy as np
from ete3 import *
import random
from sklearn.model_selection import train_test_split
from datetime import datetime

"""this file holds all data preparation methods for data given as a csv file
	NOTICE: this module is only meant to be used for training a NN for regression"""


def split_test_train(p=0.2, file_path=r"dirpath\learning_subset_1000ds.csv"):
	# p = precent of data to use as Test
	# saves TWO csv files 'Training_set' and 'Test_set'
	random.seed(datetime.now())
	testfilename = r'dirpath\Test_set.csv'
	trainfilename = r'dirpath\Training_set.csv'
	# remove old files
	if os.path.isfile(testfilename):
		os.remove(testfilename)
	if os.path.isfile(trainfilename):
		os.remove(trainfilename)

	# reads 100000 lines every time, to handle large csv files
	for chunk in pd.read_csv(file_path, chunksize=200000):

		train, test = train_test_split(chunk, test_size=p)
		train.to_csv(trainfilename, mode='a', header=False)
		test.to_csv(testfilename, mode='a', header=False)


def handle_row(row):
	x = row[6:-1]  # remove label and first columns which aren't attributes
	y = row[-1]  # take only label
	# we want to return numbers not strings, small nums so float64
	x = np.array(x, dtype=np.float64)
	y = np.array(y, dtype=np.float64)
	return x, y


def get_train_batch(file_path=r'dirpath\Training_set.csv', batch_size=32):
	# assumes train data in file_path matches the format as saved by split_test_train() uses extra csv file because
	# pandas is very slow, using f.open and f.seek are much faster, later i use csv reader to read the sampled data.
	# this is done to avoid reading entire trainingfile=file_path to memory
	samplefilename = r'dirpath\sample.csv'
	# remove old files
	if os.path.isfile(samplefilename):
		os.remove(samplefilename)

	filesize = os.stat(file_path).st_size  # size of the really big file
	x_list = []  # attribute vectors
	y_list = []  # correct likelihood values, or so i believe

	resultfile = open(samplefilename, 'w')

	for i in range(batch_size):
		offset = random.randrange(filesize)

		f = open(file_path, 'r')  # doesnt load to memory
		f.seek(offset)  # go to random position
		f.readline()  # discard - bound to be partial line
		random_line = f.readline()  # bingo!

		# extra to handle last/first line edge cases
		if len(random_line) == 0:  # we have hit the end
			f.seek(0)
			random_line = f.readline()  # so we'll grab the first line instead
		resultfile.write(random_line)
	resultfile.close()

	with open(samplefilename, newline='') as f:
		reader = csv.reader(f)
		for row in reader:
			x, y = handle_row(row)
			x_list.append(x)  # remove label and first columns which aren't attributes
			y_list.append(y)  # take only label

	return np.vstack(x_list), np.vstack(y_list)


def get_test_data():
	# generator function, to be used iteratively
	# example: 'for x,y in get_test_data():'
	filename = r'dirpath\Test_set.csv'

	with open(filename, "r") as csvfile:
		datareader = csv.reader(csvfile)
		row = next(datareader)  # yield the header row
		x, y = handle_row(row)
		yield x, y
		for row in datareader:
			x, y = handle_row(row)
			yield x, y


def main():
	file_path = r"dirpath\learning_subset_1000ds.csv"
	file_path = r'dirpath\Training_set.csv'

	# x, y = get_train_batch(file_path, 64)
	for x, y in get_test_data():
		print(x)
		print(y)
	print("done!")
	exit(0)


if __name__ == '__main__':
	main()
