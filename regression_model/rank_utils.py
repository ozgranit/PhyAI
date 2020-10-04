import sys
import os
import numpy as np
import csv
from csv import reader, writer
from pathlib import Path

parent_path = Path().resolve().parent

data_folder = parent_path / 'data'


def add_ranks(file_name):
    with open(file_name, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        scores_list = []
        first = True
        for row in csv_reader:
            if first:
                first = False
            else:
                scores_list.append(float(row[-1]))

        order = np.argsort(scores_list)
        ranks = np.argsort(order)
        ranks = [(len(ranks)-rank) for rank in ranks]

    with open(file_name, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        out = writer(open(file_name[:-4]+"_ranks" + ".csv", "w", newline=''), delimiter=",")
        i = 0
        first = True
        for row in csv_reader:
            if first:
                row.append("Rank")
                first = False
            else:
                row.append(ranks[i])
                i += 1
            out.writerow(row)
    csv_file.close()
    os.remove(csv_file.name)


def add_ranks_all_files():
    i = 1
    while True:
        file_name = data_folder / ('results/output' + str(i) + '.csv')
        try:
            add_ranks(file_name)
            i += 1
        except:
            print("done")
            break


def csv_split():
    # file_name = "learning_subset_1000ds.csv"
    file_name = "..\data/learning_all_moves_step1.csv"

    index = 1
    out = writer(open("../data/results/output" + str(index) + ".csv", "w", newline=''), delimiter=",")
    current_value = "headlines"

    index_of_path_column = 0  # 3  # notice change in index to fit new clean file (7G)

    with open(file_name, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if current_value == "first":  # if this is the first row of the big file(after headlines)
                current_value = row[index_of_path_column]

            if current_value == "headlines":  # if we are reading the headlines
                headlines = row
                out.writerow(headlines)
                current_value = "first"
            else:
                if row[index_of_path_column] == current_value:
                    out.writerow(row)
                else:
                    current_value = row[index_of_path_column]
                    index += 1
                    out = writer(open("../data/results/output" + str(index) + ".csv", "w", newline=''), delimiter=",")
                    out.writerow(headlines)
                    out.writerow(row)


if __name__ == '__main__':
    csv_split()
    add_ranks_all_files()

