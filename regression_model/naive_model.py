import csv
import numpy as np
from prepare_data import handle_row, get_test_data
from pathlib import Path

# meant to be used as a reference point for the effectiveness of other models
# this is the most naive regression solution possible


def naive_model():
	# return the mean of the train set
	Label_lst = []
	parent_path = Path().resolve().parent
	filename = parent_path / 'data/Training_set.csv'

	with open(filename, "r") as csvfile:
		datareader = csv.reader(csvfile)
		row = next(datareader)  # yield the header row
		x, y = handle_row(row)
		Label_lst.append(y)
		for row in datareader:
			x, y = handle_row(row)
			Label_lst.append(y)

	return np.mean(Label_lst)


def SmoothL1Loss(x, y):
	if abs(x-y) < 1:
		return 0.5*(x-y)**2
	else:
		return abs(x-y)-0.5


def L1Loss(x, y):
	return abs(x-y)


def test_naive_model():

	val = naive_model()
	TestLoss = []

	for x, y in get_test_data():
		loss = L1Loss(val, y)
		TestLoss.append(loss.item())
	return np.mean(TestLoss)


if __name__ == '__main__':
	print(test_naive_model())
