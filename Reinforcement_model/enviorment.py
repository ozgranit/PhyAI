import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import bio_methods


def env_reset():
	# take new tree from trainning datasets
	# make tree into matrix
	# save tree for play_action method
	# return matrix as vector numpy
	state = 0
	return torch.tensor(state)


def play_action(state, action):
	# get saved tree
	# convert action to two nodes("sp000-sp019 or N1-N20")
	# get ll of saved tree
	# use two nodes and old tree to get new tree
	# save new tree
	# make new tree into matrix
	# reward = new ll - old ll
	# return matrix and reward
	next_state = 0
	reward = 1
	return torch.tensor(next_state), reward
