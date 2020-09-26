import os
import sys
import pickle
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from prepare_data import get_train_batch, get_test_data

classfication = True
num_of_classes = 100



def plot_loss(TrainLoss, TestLoss=None):
	# plot will work with or without Testloss
	plt.clf()
	plt.xlabel('Steps')
	plt.ylabel('L1Loss, averaged over 1000 steps')
	plt.plot(range(0, 1000*len(TrainLoss), 1000), TrainLoss, label="Train-loss")
	if TestLoss is not None:
		plt.plot(TestLoss[0], TestLoss[1], 'rx',  label="Test-loss")
	plt.legend()
	plt.title("Performance")
	plt.savefig('LNN-Performance.png')


# test model on test-set using mean of loss
def test_model(model):
	TestLoss = []
	if classfication:
		criterion = nn.CrossEntropyLoss()
	else:
		criterion = nn.L1Loss()
	with torch.no_grad():
		for x, y in get_test_data():
			x = torch.tensor(x, dtype=torch.float32)
			y = torch.tensor(y, dtype=torch.float32)

			predictions = model(x.float())
			loss = criterion(predictions, y)
			TestLoss.append(loss.item())
	return np.mean(TestLoss)


def load_model_and_loss(model):
	if os.path.isfile('N_params.pkl'):
		print('Load N parameters ...')
		model.load_state_dict(torch.load('N_params.pkl'))

	# load prev Stats
	start = 0
	TrainLoss = []  # TrainLoss holds the average loss of last 100 time_steps
	TestLoss = [[], []]   # TestLoss[1] is test lost val, TestLoss[0] is test lost time_step

	TRAIN_LOSS_FILE = 'TrainLoss.pkl'
	if os.path.isfile(TRAIN_LOSS_FILE):
		with open(TRAIN_LOSS_FILE, 'rb') as f:
			TrainLoss = pickle.load(f)
			start = 1000*len(TrainLoss)
			print('Load %s ...' % TRAIN_LOSS_FILE)

	TEST_LOSS_FILE = 'TestLoss.pkl'
	if os.path.isfile(TEST_LOSS_FILE):
		with open(TEST_LOSS_FILE, 'rb') as f:
			TestLoss = pickle.load(f)
			print('Load %s ...' % TEST_LOSS_FILE)

	return start, TrainLoss, TestLoss


def save_model_and_plot(model, TrainLoss, TestLoss, time_step):
	print("Timestep %d" % (time_step,))
	# print("Train loss %f" % TrainLoss[-1])
	# print("Test loss %f" % )
	sys.stdout.flush()

	# Save the trained model
	torch.save(model.state_dict(), 'N_params.pkl')
	TRAIN_LOSS_FILE = 'TrainLoss.pkl'
	TEST_LOSS_FILE = 'TestLoss.pkl'
	# Dump statistics to pickle
	with open(TRAIN_LOSS_FILE, 'wb') as f:
		pickle.dump(TrainLoss, f)
		# print("Saved to %s" % TRAIN_LOSS_FILE)

	with open(TEST_LOSS_FILE, 'wb') as f:
		pickle.dump(TestLoss, f)
		# print("Saved to %s" % TEST_LOSS_FILE)
	print("Saved Stats")
	plot_loss(TrainLoss, TestLoss)
