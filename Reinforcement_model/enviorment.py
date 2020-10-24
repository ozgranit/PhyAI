import os
import sys
import pickle
import matplotlib.pyplot as plt

import torch
import bio_methods


def env_reset():
	# take new tree from trainning datasets
	# make tree into matrix
	# return matrix as vector numpy
	state = bio_methods.graph_from_tree()
	return torch.tensor(state)


def play_action(state, action):
	next_state = 0
	reward = 1
	return torch.tensor(next_state), reward

if __name__ == '__main__':
	env_reset()