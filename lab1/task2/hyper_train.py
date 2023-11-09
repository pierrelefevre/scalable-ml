import hopsworks
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import sys
from multiprocessing import Process
import time


def train(thread_id, X_train, X_validation, X_test, y_train, y_validation, y_test):

    log_file = open(f"log/hyper_train_{thread_id}.log", "w")

    # Create a list of models to evaluate
    models = []

    for i in range(2, 50):
        models.append((f'KNN_{i}', KNeighborsClassifier(n_neighbors=i)))

    # add regressor
    models.append(('LR', LinearRegression()))
    models.append(('SGD', SGDRegressor()))

    # evaluate each model in turn
    results = []

    # try dropping columns
    max_name = ""
    max_score = 0
    max_columns = []

    i = 0
    while True:
        np.random.seed(int(time.time() * 1000) % 2**32)
        randomized_perm = np.random.permutation(X_train.columns)[: max(
            1, int(np.random.rand() * len(X_train.columns)))]
        not_in_perm = [x for x in X_train.columns if x not in randomized_perm]

        X_train_modified = X_train.drop(not_in_perm, axis=1)
        X_validation_modified = X_validation.drop(not_in_perm, axis=1)

        # for loop to iterate over different models
        for (j, (name, model)) in enumerate(models):
            model.fit(X_train_modified, y_train.values.ravel())
            y_pred = model.predict(X_validation_modified)

            score = model.score(X_validation_modified, y_validation)
            columns = X_train_modified.columns

            results.append({"name": name, "y_pred": y_pred,
                            "score": score, "columns": columns})

            if score > max_score:
                max_score = score
                max_name = name
                max_columns = list(columns)

            log = f'({i*len(models)+j}): {name}:{score:0.3f} highest: {max_name}, {max_score:0.3f}, [{max_columns}]'
            log_file.write(log + "\n")
        i += 1
    log_file.close()


def main():
    project = hopsworks.login(project="id2223_pierrelf_emilk2")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="winequality", version=1)
    query = fg.select_all()
    feature_view = fs.get_or_create_feature_view(name="winequality",
                                                 version=1,
                                                 description="Read from winequality dataset",
                                                 labels=["quality"],
                                                 query=query)

    # remove old logfiles
    for file in os.listdir():
        if file.endswith(".log"):
            os.remove(file)

    num_threads = sys.argv[1]

    for i in range(int(num_threads)):
        X_train, X_validation, X_test, y_train, y_validation, y_test = feature_view.train_validation_test_split(
            0.2, 0.2)
        p = Process(target=train, args=(i, X_train, X_validation,
                    X_test, y_train, y_validation, y_test,))
        p.start()


if __name__ == "__main__":
    main()

# # print highest 10 scores
# results.sort(key=lambda x: x["score"], reverse=True)
# for (result) in results[:10]:
#     print(f"{result['name']}: {result['score']}")

# input("Press Enter to continue...")

# for (result) in results:
#     metrics = classification_report(y_test, result["y_pred"], output_dict=True)
#     matrix = confusion_matrix(y_test, result["y_pred"])

#     df_cm = pd.DataFrame(matrix)
#     cm = sns.heatmap(df_cm, annot=True)
#     fig = cm.get_figure()
