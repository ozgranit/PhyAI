import os
import sys
import pickle
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

import matplotlib.pyplot as plt
from prepare_data import get_train_batch, get_test_data

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


def plot_loss(TrainLoss, TestLoss=None):
	# plot will work with or without Testloss
	plt.clf()
	plt.xlabel('Steps')
	plt.ylabel('Log-Loss')
	plt.plot(range(len(TrainLoss)), TrainLoss, label="Train-loss")
	if TestLoss is not None:
		plt.plot(range(len(TestLoss)), TestLoss, label="Test-loss")
	plt.legend()
	plt.title("Performance")
	plt.savefig('LNN-Performance.png')


# test model on test-set using mean of loss
def test_model(model):
	TestLoss = []
	with torch.no_grad():
		for x, y in get_test_data():
			x = torch.tensor(x, dtype=torch.float32)
			y = torch.tensor(y, dtype=torch.float32)

			predictions = model(x.float())
			loss = nn.SmoothL1Loss()(predictions, y)
			TestLoss.append(np.log(loss.item()))
	return np.mean(TestLoss)


# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
	classname = m.__class__.__name__
	# for every Linear layer in a model..
	if classname.find('Linear') != -1:
		# get the number of the inputs
		n = m.in_features
		y = 1.0/np.sqrt(n)
		m.weight.data.uniform_(-y, y)
		m.bias.data.fill_(0)


def lnn_learning(
		lnn,
		time_steps,
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

	# initialize weights
	N.apply(weights_init_uniform_rule)

	# Check & load pretrained model
	if os.path.isfile('N_params.pkl'):
		print('Load N parameters ...')
		N.load_state_dict(torch.load('N_params.pkl'))
	######

	# Loss and Optimizer
	criterion = nn.SmoothL1Loss()  # nn.MSELoss()
	optimizer = optim.SGD(N.parameters(), lr=learning_rate)

	TrainLoss = []
	TestLoss = []

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
	LOG_EVERY_N_STEPS = 10000
	CALC_TEST_EVERY_N_STEPS = 50000

	for t in range(start, time_steps):
		x_train, labels = get_train_batch(batch_size=batch_size)
		# for x_train, labels in get_test_data():
		# not sure which one will work #1 or #2
		# 1:
		x_train = torch.tensor(x_train, dtype=torch.float32)
		labels = torch.tensor(labels, dtype=torch.float32)
		# 2:
		# x_train = Variable(x_train)
		# labels = Variable(labels)

		# Forward
		predictions = N(x_train.float())
		if t > time_steps-2:
			print(predictions[0],  labels[0])
		loss = criterion(predictions, labels)
		TrainLoss.append(np.log(loss.item()))

		# Backward + Optimize
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		if t % CALC_TEST_EVERY_N_STEPS == 0 and t > 1:
			TestLoss.append(test_model(N))

		if t % LOG_EVERY_N_STEPS == 0 and t > 1:
			print("Timestep %d" % (t,))
			# print("Train loss %f" % TrainLoss[-1])
			# print("Test loss %f" % )
			sys.stdout.flush()

			# Save the trained model
			torch.save(N.state_dict(), 'N_params.pkl')
			# Dump statistics to pickle
			with open(STATS_FILE_NAME, 'wb') as f:
				pickle.dump(TrainLoss, f)
				print("Saved to %s" % STATS_FILE_NAME)

			plot_loss(TrainLoss, TestLoss)

	return TrainLoss
