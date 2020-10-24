import csv
import random

import numpy as np
import torch
import bio_methods

from reinforcement_main import NUM_ACTIONS
from pathlib import Path

parent_path = Path().resolve().parent

parent_folder = parent_path

global current_tree
global current_msa_path
global current_likelihood


def num_to_action(n):
	assert 0 <= n < NUM_ACTIONS
	# 0 always defined as no-op, gives the model a chance to stay in place
	if n == 0:
		return -1
	# possible pairs: ('Sp000', 'Sp001')...('Sp019', 'Sp018')
	# allow duplicates ('Sp000', 'Sp001') and ('Sp001', 'Sp000')
	# 19 options for each sp times 20 taxa = 380 pairs
	first = n // 19   # // means get int from division
	second = n % 19


def set_random_msa_path():
	global current_msa_path
	rand_row = random.randit(1, 2992)
	csv_reader = csv.reader(parent_path / "data/sampled_datasets.csv", delimiter=',')
	for i, row in enumerate(csv_reader):
		if i == rand_row:
			current_msa_path = row[3]
		break

def env_reset():
	# take new tree from trainning datasets
	# make tree into matrix
	# save tree for play_action method
	# return matrix as vector numpy
	global current_tree
	global current_msa_path

	# setting a random folder from the different msa folders
	set_random_msa_path()

	tree = bio_methods.get_tree_from_msa(msa_path=current_msa_path)
	matrix = bio_methods.tree_to_matrix(tree=tree)
	current_tree = tree
	return torch.tensor(matrix)


def play_action(state, action):
	global current_likelihood
	global current_tree
	global current_msa_path

	# convert action to two nodes("sp000-sp019 or N1-N20")
	cut_name, paste_name = num_to_action(action)

	# use two nodes and old tree to get new tree
	new_tree = bio_methods.SPR_by_edge_names(current_tree, cut_name, paste_name)

	# make new tree into matrix
	next_state = bio_methods.tree_to_matrix(new_tree)

	# calculating reward
	new_likelihood = bio_methods.get_likelihood_simple(tree=new_tree, msa_path=current_msa_path)
	reward = new_likelihood - current_likelihood

	current_likelihood = new_likelihood
	current_tree = new_tree

	return next_state, reward


if __name__ == '__main__':
	env_reset()

	# for every tree folder
	# get_graph
	#