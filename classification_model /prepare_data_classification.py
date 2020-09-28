import csv
import pandas as pd
import numpy as np
from ete3 import *
import random
from sklearn.model_selection import train_test_split
from datetime import datetime
from rank_utils_classification import csv_split, add_ranks_all_files, create_big_ranked_file_from_learning_all_moves_step1
from pathlib import Path

"""this file holds all data preparation methods for data given as a csv file
	NOTICE: this module is only meant to be used for training a NN for regression"""

FEATURE_LIST = ['edge_length_prune', 'longest_branch', 'ntaxa_prunned_prune', 'pdist_average_pruned_prune',
                'tbl_pruned_prune', 'parsimony_pruned_prune',
                'longest_pruned_prune', 'ntaxa_remaining_prune', 'pdist_average_remaining_prune', 'tbl_remaining_prune',
                'parsimony_remaining_prune',
                'longest_remaining_prune', 'orig_ds_tbl', 'edge_length_rgft', 'ntaxa_prunned_rgft',
                'pdist_average_pruned_rgft', 'tbl_pruned_rgft',
                'parsimony_pruned_rgft', 'longest_pruned_rgft', 'ntaxa_remaining_rgft', 'pdist_average_remaining_rgft',
                'tbl_remaining_rgft',
                'parsimony_remaining_rgft', 'longest_remaining_rgft', 'topology_dist_between_rgft',
                'tbl_dist_between_rgft',
                'res_tree_edge_length_rgft', 'res_tree_tbl_rgft']

parent_path = Path().resolve().parent

dirpath_folder = parent_path / 'dirpath'

def split_test_train(p=0.2, file_path=dirpath_folder / 'big_file_ranked.csv'):
	# p = precent of data to use as Test
	# saves TWO csv files 'Training_set' and 'Test_set'
	random.seed(datetime.now())
	# testfilename = r'..\dirpath\Test_set.csv'
	testfilename = dirpath_folder / "Test_set.csv"
	# trainfilename = r'..\dirpath\Training_set.csv'
	trainfilename = dirpath_folder / "Training_set.csv"

	# remove old files
	if os.path.isfile(testfilename):
		os.remove(testfilename)
	if os.path.isfile(trainfilename):
		os.remove(trainfilename)

	# reads 100000 lines every time, to handle large csv files
	for chunk in pd.read_csv(file_path, chunksize=200000):
		train, test = train_test_split(chunk, test_size=p)
		train.to_csv(trainfilename, mode='a', header=False, index=False)
		test.to_csv(testfilename, mode='a', header=False, index=False)


def handle_row(row):
	# was modified to handle "learning_all_moves_step1.csv", will not work on other files
	x = row[1:-2]  # remove label and first column which aren't attributes
	y = row[-1]  # take only label
	# replace '' with '0'
	x = ['0' if a == '' else a for a in x]
	# we want to return numbers not strings, small nums so float64
	x = np.array(x, dtype=np.float64)
	y = np.array(y, dtype=np.float64)
	return x, y


def get_train_batch(file_path=dirpath_folder / 'Training_set.csv', batch_size=32):
	# assumes train data in file_path matches the format as saved by split_test_train() uses extra csv file because
	# pandas is very slow, using f.open and f.seek are much faster, later i use csv reader to read the sampled data.
	# this is done to avoid reading entire trainingfile=file_path to memory
	# samplefilename = r'..\dirpath\sample.csv'
	samplefilename = dirpath_folder / "sample.csv"
	# remove old files
	if os.path.isfile(samplefilename):
		os.remove(samplefilename)

	x_list = []  # attribute vectors
	y_list = []  # correct likelihood values, or so i believe

	resultfile = open(samplefilename, 'w')

	for i in range(batch_size):
		filesize = os.stat(file_path).st_size  # size of the specific file
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
		f.close()
	resultfile.close()

	with open(samplefilename, newline='') as f:
		reader = csv.reader(f)
		for row in reader:
			x, y = handle_row(row)
			x_list.append(x)
			y_list.append(y)

	return np.vstack(x_list), np.vstack(y_list)


def get_test_data():
	# generator function, to be used iteratively
	# example: 'for x,y in get_test_data():'
		# filename = r'..\dirpath\class_results\test\output'+str(file_num)+'_ranks'
		filename = dirpath_folder / "Test_set.csv"
		with open(filename, "r") as csvfile:
			datareader = csv.reader(csvfile)
			row = next(datareader)  # yield the header row
			x, y = handle_row(row)
			yield x, y
			for row in datareader:
				x, y = handle_row(row)
				yield x, y


def clean_all_step_file():
	# method for cleaning the 12 gb file
	# to be called once, after extraction
	# new_file = r'..\dirpath\new_set.csv'
	new_file = dirpath_folder / 'new_set.csv'
	# file_path = '../dirpath/learning_all_moves_step1.csv'
	file_path = dirpath_folder / 'learning_all_moves_step1.csv'

	if not os.path.isfile(file_path):
		print("No learning_all_moves_step1.csv found in dirpath")
		exit(0)

	# remove old files
	if os.path.isfile(new_file):
		os.remove(new_file)

	# save header only once to new csv
	first_save = True
	for chunk in pd.read_csv(file_path, chunksize=200000):
		print("running")
		for column in chunk:
			if not(column == 'd_ll_merged' or column == 'path' or column in FEATURE_LIST):
				chunk.drop(column, axis=1, inplace=True)
		if first_save:
			chunk.to_csv(new_file, mode='a', header=True, index=False)
			first_save = False
		else:
			chunk.to_csv(new_file, mode='a', header=False, index=False)

def main():
	file_path = r"..\dirpath\learning_subset_1000ds.csv"
	file_path = r'..\dirpath\Training_set.csv'

	# x, y = get_train_batch(file_path, 64)
	for x, y in get_test_data():
		print(x)
		print(y)
	print("done!")
	exit(0)


if __name__ == '__main__':
	# clean_all_step_file()

	create_big_ranked_file_from_learning_all_moves_step1()
	split_test_train()




#get_train_batch()


