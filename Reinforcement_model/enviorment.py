import numpy as np
import torch
import bio_methods

from reinforcement_main import NUM_ACTIONS


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


def env_reset():
	# take new tree from trainning datasets
	# make tree into matrix
	# save tree for play_action method
	# return matrix as vector numpy
	state = np.array([0, 1,
	                  1, 0])
	return torch.tensor(state).float()


def play_action(state, action):
	# get saved tree
	# convert action to two nodes("sp000-sp019 or N1-N20")
	# get ll of saved tree
	# use two nodes and old tree to get new tree
	# save new tree
	# make new tree into matrix
	# reward = new ll - old ll
	# return matrix and reward
	next_state = env_reset()
	reward = next_state[action - 1].item()
	return next_state, reward
