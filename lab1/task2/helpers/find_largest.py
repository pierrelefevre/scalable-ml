# get all .log files in dir
import os
files = os.listdir()
log_files = [file for file in files if file.endswith(".log")]

# read the last row from each hyper_train_n.log file
last_rows = []
for file in log_files:
    with open(file, "r") as f:
        last_rows.append(f.readlines()[-1].replace("\n", "").split("highest: ")[1])

last_rows.sort(key=lambda x: x.split(",")[1], reverse=True)

for row in last_rows:
    print(row)
