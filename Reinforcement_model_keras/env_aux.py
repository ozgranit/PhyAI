import csv
import numpy as np
from rein_model import INPUT_SIZE, NUM_NODES

from pathlib import Path

parent_path = Path().resolve().parent
parent_folder = parent_path / "reinforcement_data"

"""we maintain 3 different tree formats for our env, string is the most general one, 
we convert to ete or bio when needed,
ete - for most bio- methods - pruning regrafting and such,
bio - for converting tree to matrix
"""


def n_from_int(n):
	# this method translates int to internal node N{}
	assert 20 <= n <= 38
	n -= 19
	if n < 10:
		return 'N00' + str(n)
	return 'N0' + str(n)


def sp_from_int(n):
	# this method translates int to leaf node Sp{}
	assert 0 <= n <= 19
	if n < 10:
		return 'Sp00' + str(n)
	return 'Sp0' + str(n)


def sp_or_n(idx):
	if idx < 20:
		return sp_from_int(idx)
	else:
		return n_from_int(idx)


def get_action_matrix():
	# 2D list, 39*39
	matrix = [[0]*NUM_NODES]*NUM_NODES
	# first 20 are Sp.. last 19 are N.
	for row in range(NUM_NODES):
		for col in range(NUM_NODES):
			first = sp_or_n(row)
			second = sp_or_n(col)
			matrix[row][col] = (first, second)
	return matrix


def set_random_msa_path():
	rand_row = np.random.randint(1, 2992)
	path = parent_folder / "data/sampled_datasets.csv"
	with open(path, "r") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for i, row in enumerate(csv_reader):
			if i == rand_row:
				path = row[3]
				current_msa_path = path[1:]
				return current_msa_path
