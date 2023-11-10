# get all .log files in dir
path = "../log/"

import os
files = os.listdir(path)
log_files = [file for file in files if file.endswith(".txt")]


# read the last row from each hyper_train_n.log file
last_rows = []
for file in log_files:
    with open(path + file, "r") as f:
        last_rows.append(f.readlines()[-1].replace("\n", "").split(" bestest: ")[1])

last_rows.sort(key=lambda x: x.split("score:")[1].split("mse:")[0], reverse=True)

for row in last_rows:
    print(row)
