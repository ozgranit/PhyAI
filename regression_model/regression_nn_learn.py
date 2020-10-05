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
from lnn_utils import plot_loss, test_model, load_model_and_loss, save_model_and_plot


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

SavedState = namedtuple("SavedState", "state_dict timestep stats")
print('******* Running on {} *******'.format('CUDA' if USE_CUDA else 'CPU'))


class Variable(autograd.Variable):
	def __init__(self, data, *args, **kwargs):
		if USE_CUDA:
			data = data.cuda()
		super(Variable, self).__init__(data, *args, **kwargs)


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

	################################
	# Check & load pretrained model and loss
	# TestLoss[1] is test lost val, TestLoss[0] is test lost time_step
	# TrainLoss holds the average loss of last 100 time_steps
	start, TrainLoss, TestLoss = load_model_and_loss(N)
	last_1000_train_loss = []  # list of length up to 1000, will be lost in saving and loading
	################################

	# Loss and Optimizer
	criterion = nn.L1Loss()
	optimizer = optim.SGD(N.parameters(), lr=learning_rate)
	# every step_size we update new_lr = old_lr*gamma
	# NOTICE in loading we do NOT load the last lr used, so adjust lr manually before starting
	#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=(time_steps/10), gamma=0.2)

	LOG_EVERY_N_STEPS = 100000
	CALC_TEST_EVERY_N_STEPS = 500000

	for t in range(start, time_steps):
		###################
		# ACTUAL TRAINING #
		###################
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
		loss = criterion(predictions, labels)
		last_1000_train_loss.append(loss.item())

		# Backward + Optimize
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		# update lr
		#scheduler.step()
		#######################
		# END ACTUAL TRAINING #
		#######################

		##########################
		# STATISTICS AND LOGGING #
		##########################
		if len(last_1000_train_loss) >= 1000:
			TrainLoss.append(np.mean(last_1000_train_loss))
			last_1000_train_loss = []

		if t % CALC_TEST_EVERY_N_STEPS == 0 and t > 1:
			TestLoss[0].append(t)
			TestLoss[1].append(test_model(N))

		if t % LOG_EVERY_N_STEPS == 0 and t > 1:
			#print(scheduler.get_lr())
			print(TrainLoss[-1])
			save_model_and_plot(N, TrainLoss, TestLoss, t)

	# calc test loss and plot final result
	# TestLoss[0].append(time_steps)
	# TestLoss[1].append(test_model(N))
	plot_loss(TrainLoss, TestLoss)
	return TrainLoss, TestLoss
