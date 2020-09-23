import os
import sys
import pickle
import numpy as np

import torch
import torch.autograd as autograd

import matplotlib.pyplot as plt
from lnn_utils import load_model_and_loss


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

SavedState = namedtuple("SavedState", "state_dict timestep stats")
print('******* Running on {} *******'.format('CUDA' if USE_CUDA else 'CPU'))


def model_predict(model):

	with torch.no_grad():
		for x, y in get_test_data():
			x = torch.tensor(x, dtype=torch.float32)
			y = torch.tensor(y, dtype=torch.float32)

			predictions = model(x.float())
			loss = criterion(predictions, y)
			TestLoss.append(loss.item())
	return np.mean(TestLoss)


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

	LOG_EVERY_N_STEPS = 5000
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


	plot_loss(TrainLoss, TestLoss)
	return predictions
