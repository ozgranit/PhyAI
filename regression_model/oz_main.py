import os
import sys
from regression_model import LNN
from regression_nn_learn import plot_loss, lnn_learning
from prepare_data import split_test_train
from naive_model import test_naive_model
import numpy as np

BATCH_SIZE = 32*2
LEARNING_RATE = 1e-5


def main(time_steps):

	TrainLoss, TestLoss = lnn_learning(
		lnn=LNN,
		time_steps=time_steps,
		learning_rate=LEARNING_RATE,
		batch_size=BATCH_SIZE,
	)
	naive_loss = test_naive_model()
	# we don't always bother filling TestLoss
	try:
		print("LNN Best Loss: %f" % min(TestLoss[1]))
		print("Naive Model Loss: %f" % naive_loss)
		if naive_loss < min(TestLoss[1]):
			print("Naive Model did better.")
		else:
			print("LNN did better.")

		idx = TestLoss[1].index(min(TestLoss[1]))
		print("LNN Best Loss after %d Steps" % TestLoss[0][idx])
	except ValueError:
		print("No TestLoss, Moving on")
	# plot_loss(TrainLoss, TestLoss)


if __name__ == '__main__':
	# split_test_train(file_path=r"dirpath\model to overfit.csv")
	# split_test_train(p=0.5, file_path=r"dirpath\example-oz.csv")
	# split_test_train(file_path=r"dirpath\learning_subset_1000ds.csv")
	# split_test_train(file_path=r"dirpath\learning_all_moves_step1.csv")

	time_steps = 1000001
	# Run training
	main(time_steps)
	#
	# if os.path.isfile('N_params.pkl'):
	# 	os.remove('N_params.pkl')
	# if os.path.isfile('statistics.pkl'):
	# 	os.remove('statistics.pkl')
