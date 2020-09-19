import csv
import sys
from csv import reader, writer


if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
   # file_name = "learning_subset_1000ds.csv"
     file_name = "learning_all_moves_step1.csv"

index = 1
out = writer(open("results/output" + str(index) + ".csv", "w"), delimiter=",")
current_value = "headlines"

index_of_path_column = 3 # notice that for the small file the index is 2

with open(file_name, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if current_value == "first":#if this is the first row of the big file(after headlines)
            current_value = row[index_of_path_column]

        if current_value == "headlines":#if we are reading the headlines
            headlines = row
            out.writerow(headlines)
            current_value = "first"
        else:
            if row[index_of_path_column] == current_value:
                out.writerow(row)
            else:
                current_value = row[index_of_path_column]
                index += 1
                out = writer(open("results/output" + str(index) + ".csv", "w", newline=''), delimiter=",")
                out.writerow(headlines)
                out.writerow(row)



