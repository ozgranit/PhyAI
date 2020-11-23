import csv
import random
import torch
import bio_methods

from pathlib import Path

parent_path = Path().resolve().parent
parent_folder = parent_path / "reinforcement_data"

global current_ete_tree
global current_bio_tree
global current_tree_str
global current_msa_path
global current_likelihood
global likelihood_params


def sp_from_int(n):
	assert 0 <= n <= 19
	if n < 10:
		return 'Sp00'+str(n.item())
	return 'Sp0'+str(n.item())


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
	rand_row = random.randint(1, 2992)
	path = parent_folder / "data/sampled_datasets.csv"
	with open(path, "r") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for i, row in enumerate(csv_reader):
			if i == rand_row:
				path = row[3]
				current_msa_path = path[1:]
				break


def env_reset():
	# take new tree from trainning datasets
	# make tree into matrix
	# save tree for play_action method
	# return matrix as vector numpy
	global current_ete_tree
	global current_tree_str
	global current_bio_tree
	global current_msa_path
	global likelihood_params
	global current_likelihood

	# setting a random folder from the different msa folders
	set_random_msa_path()
	likelihood_params = bio_methods.calc_likelihood_params(current_msa_path)
	ete_tree, bio_tree, tree_str = bio_methods.get_tree_from_msa(msa_path=current_msa_path)

	current_likelihood = bio_methods.get_likelihood_simple(tree_str, current_msa_path, likelihood_params)
	matrix = bio_methods.tree_to_matrix(bio_tree)

	current_ete_tree = ete_tree
	current_bio_tree = bio_tree
	current_tree_str = tree_str

	return torch.tensor(matrix)


def play_action(state, action):
	global current_likelihood
	global current_ete_tree
	global current_msa_path

	# convert action to two nodes("sp000-sp019 or N1-N20")
	cut_name, paste_name = num_to_action(action)

	if cut_name is None and paste_name is None:
		return state, 0

	# use two nodes and old tree to get new tree
	print(current_ete_tree.get_ascii(show_internal=True))
	print(cut_name, paste_name)
	new_ete_tree = bio_methods.SPR_by_edge_names(current_ete_tree, cut_name, paste_name)

	# make new tree into matrix
	next_state = bio_methods.tree_to_matrix(new_ete_tree)

	# calculating reward
	new_likelihood = bio_methods.get_likelihood_simple(tree=new_ete_tree, msa_path=current_msa_path, params=likelihood_params)
	reward = new_likelihood - current_likelihood

	current_likelihood = new_likelihood
	current_ete_tree = new_ete_tree

	return next_state, reward
