import os
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import matplotlib.pyplot as plt
from prepare_data import get_train_batch

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

SavedState = namedtuple("SavedState", "state_dict timestep stats")
# STATISTICS_FILE_PATH = 'statistics.pkl'
print('******* Running on {} *******'.format('CUDA' if USE_CUDA else 'CPU'))


class Variable(autograd.Variable):
	def __init__(self, data, *args, **kwargs):
		if USE_CUDA:
			data = data.cuda()
		super(Variable, self).__init__(data, *args, **kwargs)


def plot_loss(TrainLoss):
	plt.clf()
	plt.xlabel('Epochs')
	plt.ylabel('Train Loss')
	num_items = len(TrainLoss)
	plt.plot(range(num_items), TrainLoss, label="loss")
	plt.legend()
	plt.title("Performance")
	plt.savefig('LNN-Performance.png')


def calc_mse(predictions, labels):
	x = (predictions-labels)**2
	return torch.mean(x)


def lnn_learning(
		lnn,
		num_epochs,
		learning_rate=1e-3,
		batch_size=32,
):
	STATS_FILE_NAME = 'statistics.pkl'

	###############
	# BUILD MODEL #
	###############

	x_list, y_list = get_train_batch(batch_size=1)  # for length of x vector
	if USE_CUDA:
		N = lnn(in_features=len(x_list[0]), output=1).cuda()
	else:
		N = lnn(in_features=len(x_list[0]), output=1)

	# Check & load pretrained model
	if os.path.isfile('N_params.pkl'):
		print('Load N parameters ...')
		N.load_state_dict(torch.load('N_params.pkl'))
	######

	# Loss and Optimizer
	criterion = nn.MSELoss()
	optimizer = optim.SGD(N.parameters(), lr=learning_rate)

	TrainLoss = []

	# load prev Stats
	start = 0
	if os.path.isfile(STATS_FILE_NAME):
		with open(STATS_FILE_NAME, 'rb') as f:
			TrainLoss = pickle.load(f)
			start = len(TrainLoss)
			print('Load %s ...' % STATS_FILE_NAME)

	###############
	# RUN TRAINING#
	###############
	LOG_EVERY_N_STEPS = 1000

	for epoch in range(start, num_epochs):
		x_train, labels = get_train_batch(batch_size=batch_size)
		# not sure which one will work #1 or #2
		# 1:
		x_train = torch.tensor(x_train, dtype=torch.float32)
		labels = torch.tensor(labels, dtype=torch.float32)
		# 2:
		# x_train = Variable(x_train)
		# labels = Variable(labels)

		# Forward
		predictions = N(x_train.float())
		if epoch > num_epochs-2:
			print(predictions[0],  labels[0])
		loss = criterion(predictions, labels).clamp(-100000, 100000)
		TrainLoss.append(loss.item())

		# Backward + Optimize
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		if epoch % LOG_EVERY_N_STEPS == 0 and epoch > 1:
			print("Epoch %d" % (epoch,))
			# print("Train loss %f" % TrainLoss[-1])
			# print("Test loss %f" % )
			sys.stdout.flush()

			# Save the trained model
			torch.save(N.state_dict(), 'N_params.pkl')
			# Dump statistics to pickle
			with open(STATS_FILE_NAME, 'wb') as f:
				pickle.dump(TrainLoss, f)
				print("Saved to %s" % STATS_FILE_NAME)

			plot_loss(TrainLoss)

	return TrainLoss
