import csv
import sys
from csv import reader, writer

data = list(reader(open("learning_subset_1000ds.csv", "r"), delimiter=","))
index = 1
out = writer(open("output" + str(index) + ".csv", "w"), delimiter=",")

current_value = "x"
if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    file_name = "learning_subset_1000ds.csv"

with open(file_name, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if current_value == "x":
            current_value = ""
            continue
        if current_value == "":
            current_value = row[2]
        if row[2] == current_value:
            out.writerow(row)
        else:
            current_value = row[2]
            index += 1
            out = writer(open("results/output" + str(index-1) + ".csv", "w"), delimiter=",")
            out.writerow(row)
