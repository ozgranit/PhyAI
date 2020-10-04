import csv
import os
import sys
import pickle
import numpy as np

import torch
from pathlib import Path

import matplotlib.pyplot as plt
from prepare_data import get_train_batch, FEATURE_LIST
from lnn_utils import load_model_and_loss
from regression_model import LNN

parent_path = Path().resolve().parent

data_folder = parent_path / 'data'

USE_CUDA = torch.cuda.is_available()


def get_model(pkl_file='N_params.pkl'):
	if USE_CUDA:
		model = LNN(in_features=len(FEATURE_LIST), output=1).cuda()
	else:
		model = LNN(in_features=len(FEATURE_LIST), output=1)

	if os.path.isfile(pkl_file):
		print('Load N parameters ...')
		model.load_state_dict(torch.load('N_params.pkl'))
	else:
		print('Missing file to Load model')
		exit()
	return model


def model_predict(model, x):
	with torch.no_grad():
		x = torch.tensor(x, dtype=torch.float32)
		predictions = model(x.float())
	predictions = predictions.numpy()
	return predictions


def handle_row(row):
	# was modified to handle "outputi_ranks.csv", will not work on other files
	x = row[1:-2]  # remove label, ranks and first column which aren't attributes
	rank = row[-1]  # take only rank
	# replace '' with '0'
	x = ['0' if a == '' else a for a in x]
	# we want to return numbers not strings, small nums so float64
	x = np.array(x, dtype=np.float64)
	rank = int(rank)
	return x, rank


def get_x_and_ranks(file_name):
	# matches format of outputi_ranks as taken from the clean 'learning_all_moves_step1'
	x_list = []  # attribute vectors
	r_list = []  # correct ranking values
	with open(file_name, newline='') as f:
		reader = csv.reader(f)
		next(reader)  # skip first line, meaning headlines
		for row in reader:
			x, r = handle_row(row)
			x_list.append(x)
			r_list.append(r)
	return np.vstack(x_list), r_list


# returns the true ranking of the tree that was predicted 1st by the model.
# assumes predictions[i] matches with rankings[i]
def best_predicted_ranking(predictions, rankings):
	argmax = np.argmax(predictions)
	# second return value is the percentile of the result
	return rankings[argmax], (rankings[argmax]-1)/len(rankings)


def test_best_predicted_ranking():
	a = [40, 20, 3.2, 0, -17]
	b = [1, 2, 3, 4, 5]
	res = best_predicted_ranking(a, b)
	assert (res == 1)


#  returns the ranking the model predicted for the (true) best tree
def best_empirically_ranking(predictions, rankings):
	bestidx = np.argmin(rankings)
	order = np.argsort(predictions)
	model_ranks = np.argsort(order)
	model_ranks = [(len(model_ranks) - rank) for rank in model_ranks]

	return model_ranks[bestidx]


def test_best_empirically_ranking():
	a = [3.2, 40, 0, 20, -17]
	b = [3, 1, 4, 2, 5]
	res = best_empirically_ranking(a, b)
	assert (res == 1)


def handle_file(model, filename):
	x, ranks = get_x_and_ranks(filename)
	predictions = model_predict(model, x)
	res1, percentile_res1 = best_predicted_ranking(predictions, ranks)
	res2 = best_empirically_ranking(predictions, ranks)
	return res1, res2, percentile_res1


def add_col_to_csv(old_file, new_file, col, col_headline):
	reader = csv.reader(open(old_file, 'r'))
	writer = csv.writer(open(new_file, 'w', newline=''))
	headers = next(reader)
	headers.append(col_headline)
	writer.writerow(headers)
	for i, row in enumerate(reader):
		row.append(col[i])
		writer.writerow(row)


def get_results():
	model = get_model()
	true_rank_of_best_by_model = []
	precentile_of_best_by_model = []
	best_tree_rank_by_model = []

	for i in range(1, 6063):
		filename = data_folder / ("results/output" + str(i) + "_ranks.csv")
		res1, res2, percentile_res1 = handle_file(model, filename)
		true_rank_of_best_by_model.append(res1)
		precentile_of_best_by_model.append(percentile_res1)
		best_tree_rank_by_model.append(res2)

	mean1 = np.mean(true_rank_of_best_by_model)
	mean2 = np.mean(best_tree_rank_by_model)
	per = np.mean(precentile_of_best_by_model)
	print("The tree the model predicts as best is #{:.2f} (on average)".format(mean1))
	print("The tree the model predicts as best is at the top %{:.2f} (on average)".format(per*100))
	print("The best tree is ranked #{:.2f} (on average) by the model".format(mean2))
	return mean1, mean2


if __name__ == '__main__':
	get_results()
	print("done")
