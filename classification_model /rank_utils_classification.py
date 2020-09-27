import os
import numpy as np
import csv
from csv import reader, writer
from classification_model import num_of_classes
from pathlib import Path


dirpath_folder = Path("../dirpath")

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
        add_ranks_threshold(ranks, threshold=num_of_classes)

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

def add_ranks_threshold(ranks, threshold):
    for i in range(len(ranks)):
        if ranks[i] > threshold:
            ranks[i] = threshold



def add_ranks_all_files(test=False):
    i = 1
    while True:
        if test:
            # file_name = "../dirpath/class_results/test/output" + str(i) + ".csv"
            file_name = dirpath_folder / ('class_results/test/output'+str(i)+'.csv')
        else:
            # file_name = "../dirpath/class_results/train/output" + str(i) + ".csv"
            file_name = dirpath_folder / ('class_results/train/output'+str(i)+'.csv')

        try:
            add_ranks(file_name)
            i += 1
        except:
            print("done")
            return i
            break


def csv_split(file_name="../dirpath/learning_all_moves_step1.csv", test=False):
    # file_name = "learning_subset_1000ds.csv"


    index = 1
    if test:
        # out = writer(open("../dirpath/class_results/test/output" + str(index) + ".csv", "w", newline=''), delimiter=",")
        out = writer(open(dirpath_folder / ('class_results/test/output'+str(index)+'.csv'), "w", newline=''), delimiter=",")
    else:
        # out = writer(open("../dirpath/class_results/train/output" + str(index) + ".csv", "w", newline=''), delimiter=",")
        out = writer(open(dirpath_folder / ('class_results/train/output'+str(index)+'.csv'), "w", newline=''), delimiter=",")


    current_value = "headlines"

    index_of_path_column = 0  # 3  # notice change in index to fit new clean file (7G)
    flag = True


    with open(file_name, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if (flag):
                print("Shoko")
                flag=False
            else:
                print("Banana")
                flag=True
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
                    if test:
                        # out = writer(open("../dirpath/class_results/test/output" + str(index) + ".csv", "w", newline=''), delimiter=",")
                        out = writer(
                            open(dirpath_folder / ('class_results/test/output' + str(index) + '.csv'), "w", newline=''),
                            delimiter=",")

                    else:
                        # out = writer(open("../dirpath/class_results/train/output" + str(index) + ".csv", "w", newline=''), delimiter=",")
                        out = writer(open(dirpath_folder / ('class_results/train/output' + str(index) + '.csv'), "w",
                                          newline=''), delimiter=",")

                    out.writerow(headlines)
                    out.writerow(row)


if __name__ == '__main__':
    csv_split()
    add_ranks_all_files()

