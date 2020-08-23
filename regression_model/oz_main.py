import os
from regression_model import LNN
from regression_nn_learn import plot_loss, lnn_learning
from prepare_data import split_test_train

BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def main(time_steps):
	# Stats - a list to hold all results
	Stats = lnn_learning(
		lnn=LNN,
		time_steps=time_steps,
		learning_rate=LEARNING_RATE,
		batch_size=BATCH_SIZE,
	)

	plot_loss(Stats)


if __name__ == '__main__':
	# split_test_train(file_path=r"dirpath\model to overfit.csv")
	# split_test_train(p=0.5, file_path=r"dirpath\example-oz.csv")
	# split_test_train(file_path=r"dirpath\learning_subset_1000ds.csv")
	time_steps = 100000
	# Run training
	main(time_steps)

	# if os.path.isfile('N_params.pkl'):
	# 	os.remove('N_params.pkl')
	# if os.path.isfile('statistics.pkl'):
	# 	os.remove('statistics.pkl')
