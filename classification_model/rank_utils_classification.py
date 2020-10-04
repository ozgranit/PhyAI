import os
import numpy as np
import csv
from csv import reader, writer
from classification_model import num_of_classes
from pathlib import Path

parent_path = Path().resolve().parent

data_folder = parent_path / 'data'

def csv_split(file_name=data_folder / 'learning_all_moves_step1.csv'):
    index = 1
    out_file_path = data_folder / ('class_results/output' + str(index) + '.csv')
    out_file = open(out_file_path, "w", newline='')
    out = writer(out_file, delimiter=",")
    current_value = "headlines"

    index_of_path_column = 0  # 3  # notice change in index to fit new clean file (7G)

    flag = True


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
                    out_file.close()
                    out_file_path = data_folder / ('class_results/output' + str(index) + '.csv')
                    out_file = open(out_file_path, "w", newline='')
                    out = writer(out_file, delimiter=",")

                    out.writerow(headlines)
                    out.writerow(row)


def add_ranks(file_name, out_writer):
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
        add_ranks_threshold(ranks, threshold=num_of_classes)
        csv_file.close()

    with open(file_name, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        first = True
        for row in csv_reader:
            if first:
                first = False
            else:
                row.append(ranks[i])
                i += 1
                out_writer.writerow(row)
    csv_file.close()
    os.remove(csv_file.name)

def add_ranks_threshold(ranks, threshold):
    for i in range(len(ranks)):
        if ranks[i] > threshold:
            ranks[i] = threshold



def add_ranks_all_files(output_file_path):
    out_file = open(output_file_path, "w", newline='')
    out = writer(out_file, delimiter=",")

    file_name = data_folder / ('class_results/output1.csv')
    with open(file_name, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row.append("Rank")
            out.writerow(row)  # appending headlines with a new column
            break
        csv_file.close()

    i = 1
    while True:
        # file_name = "../data/class_results/train/output" + str(i) + ".csv"
        file_name = data_folder / ('class_results/output' + str(i) + '.csv')
        try:
            add_ranks(file_name, out)
            i += 1
        except:
            print("done")
            return i
            break

def create_big_ranked_file_from_learning_all_moves_step1():
    csv_split()
    add_ranks_all_files(data_folder / 'big_file_ranked.csv')



if __name__ == '__main__':
   csv_split()
   # add_ranks_all_files()

