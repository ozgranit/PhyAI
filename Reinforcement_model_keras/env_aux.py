import csv
import random
import torch
import bio_methods

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
	if n < 1 or n > 18:
		return None
	if n < 10:
		return 'N00' + str(n.item())
	return 'N0' + str(n.item())


def sp_from_int(n):
	# this method translates int to leaf node Sp{}
	assert 0 <= n <= 19
	if n < 10:
		return 'Sp00' + str(n.item())
	return 'Sp0' + str(n.item())


def num_to_action(n):
	assert 0 <= n < 400
	# 0 always defined as no-op, gives the model a chance to stay in place
	if n == 0:
		return None, None
	# possible pairs: ('Sp000', 'Sp001')...('Sp019', 'Sp018')
	# allow duplicates ('Sp000', 'Sp001') and ('Sp001', 'Sp000')
	# 19 options for each sp times 20 taxa = 380 pairs
	first = n // 20  # // means get int from division
	second = n % 20
	if first == second:
		return None, None  # no pairs of doubles allowed
	return sp_from_int(first), sp_from_int(second)


def set_random_msa_path():
	rand_row = random.randint(1, 2992)
	path = parent_folder / "data/sampled_datasets.csv"
	with open(path, "r") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for i, row in enumerate(csv_reader):
			if i == rand_row:
				path = row[3]
				current_msa_path = path[1:]
				return current_msa_path

