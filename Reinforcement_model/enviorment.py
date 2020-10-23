import os
import sys
import pickle
import matplotlib.pyplot as plt

import torch
import bio_methods


def env_reset():
	state = 0
	return torch.tensor(state)


def play_action(state, action):
	next_state = 0
	reward = 1
	return torch.tensor(next_state), reward
