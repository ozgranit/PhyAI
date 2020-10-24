import numpy as np
import torch
import bio_methods


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
	str1, str2 = num_to_action(action)
	if str1 is None and str2 is None:
		# reached no-op, do nothing and reward=0
		return state, 0
	# save new tree
	# make new tree into matrix
	# reward = new ll - old ll
	# return matrix and reward
	next_state = env_reset()
	reward = next_state[action - 1].item()
	return next_state, reward
