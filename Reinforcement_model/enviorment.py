import csv
import random

import numpy as np
import torch
import bio_methods

<<<<<<< HEAD
from reinforcement_main import NUM_ACTIONS
from pathlib import Path

parent_path = Path().resolve().parent

parent_folder = parent_path / "reinforcement_data"

global current_tree
global current_msa_path
global current_likelihood
global likelihood_params


def sp_from_int(n):
	assert 0 <= n <= 19
	if n < 10:
		return 'Sp00'+str(n)
	return 'Sp0'+str(n)

def num_to_action(n):
	assert 0 <= n < 400
	# 0 always defined as no-op, gives the model a chance to stay in place
	if n == 0:
		return None, None
	# possible pairs: ('Sp000', 'Sp001')...('Sp019', 'Sp018')
	# allow duplicates ('Sp000', 'Sp001') and ('Sp001', 'Sp000')
	# 19 options for each sp times 20 taxa = 380 pairs
	first = n // 20   # // means get int from division
	second = n % 20
	if first == second:
		return None, None   # no pairs of doubles allowed
	return sp_from_int(first), sp_from_int(second)


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
	global likelihood_params
	global current_likelihood

	# setting a random folder from the different msa folders
	set_random_msa_path()
	likelihood_params = bio_methods.calc_likelihood_params(current_msa_path)
	tree = bio_methods.get_tree_from_msa(msa_path=current_msa_path)
	current_likelihood = bio_methods.get_likelihood_simple(tree, current_msa_path, likelihood_params)
	current_tree = tree
	matrix = bio_methods.tree_to_matrix(tree=tree)
	return torch.tensor(matrix)


def play_action(state, action):
	global current_likelihood
	global current_tree
	global current_msa_path

	# convert action to two nodes("sp000-sp019 or N1-N20")
	cut_name, paste_name = num_to_action(action)

	if cut_name is None and paste_name is None:
		return state, 0

	# use two nodes and old tree to get new tree
	new_tree = bio_methods.SPR_by_edge_names(current_tree, cut_name, paste_name)

	# make new tree into matrix
	next_state = bio_methods.tree_to_matrix(new_tree)

	# calculating reward
	new_likelihood = bio_methods.get_likelihood_simple(tree=new_tree, msa_path=current_msa_path, params=likelihood_params)
	reward = new_likelihood - current_likelihood

	current_likelihood = new_likelihood
	current_tree = new_tree

	return next_state, reward


if __name__ == '__main__':
	env_reset()

	# for every tree folder
	# get_graph
	#