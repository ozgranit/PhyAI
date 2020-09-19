import csv


import os
import numpy as np
from csv import reader, writer

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

        ranks = np.argsort(scores_list)

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

i=1
while True:
    file_name = "results/output"+str(i)+".csv"
    try:
        add_ranks(file_name)
        i+=1
    except:
        print("done")
        break





