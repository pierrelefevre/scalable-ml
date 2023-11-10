import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import hopsworks
import os 
import time
import numpy as np
import sys
from multiprocessing import Process


def evaluate_model(name, params, model, X_train, y_train, X_validation, y_validation):
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train.values.ravel())
    best_model = grid_search.best_estimator_
    score = best_model.score(X_validation, y_validation)
    y_pred = best_model.predict(X_validation)
    mse = mean_squared_error(y_validation, y_pred)
    return score, mse, grid_search.best_params_

def train(thread_id, X_train, X_validation, X_test, y_train, y_validation, y_test):
    
    log_file = open(f'log/log_{thread_id}.txt', 'w')
    
    # open file as stdout
    # Create a list of models to evaluate
    models = [
        # Regressors
        ("LinReg  ", LinearRegression(), {}),
        ("PolyReg ", Pipeline([
            ("poly_features", PolynomialFeatures()),
            ("linear_regression", LinearRegression())
        ]), {'poly_features__degree': [1, 2, 3]}),
        ("RFRegres", RandomForestRegressor(), {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}),
        ("SVR     ", SVR(), {'C': [1, 10], 'gamma': ['scale', 'auto']}),
        ("DTreeReg", DecisionTreeRegressor(), {'max_depth': [None, 10, 20], 'min_samples_split': [2, 10]}),
        ("GBoostRe", GradientBoostingRegressor(), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}),
        ("LogRegre", LogisticRegression(), {'C': [1, 10], 'penalty': ['l1', 'l2']}),

        # Classifiers
        ("KNN     ", KNeighborsClassifier(), {'n_neighbors': list(range(2, 50)), 'weights': ['uniform', 'distance']}),
        ("RFClassi", RandomForestClassifier(), {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}),
        ("GBoostCl", GradientBoostingClassifier(), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}),
        ("DTreeCla", DecisionTreeClassifier(), {'max_depth': [None, 10, 20]}),
        ("ExtraTre", ExtraTreesClassifier(), {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}),
    ]

    max_name = ""
    max_score = 0
    min_mse = 0
    max_columns = []
    max_params = {}

    i = 0
    while True:
        randomized_perm, X_train_modified, X_validation_modified = get_randomized_permutation(X_train, X_validation)

        # for loop to iterate over different models
        for name, model, params in models:
            
            score, mse, best_params = evaluate_model(name, params, model, X_train_modified, y_train, X_validation_modified, y_validation)

            if score > max_score:
                max_score = score
                min_mse = mse
                max_name = name
                max_columns = randomized_perm
                max_params = best_params

            log = f'({i}): {name} ({best_params}) score: {score:0.3f} mse: {mse:.2f} bestest: {max_name} ({max_params}) score: {max_score:0.3f} mse: {min_mse:0.3f} [{list(max_columns)}]'
            log_file.write(log + '\n')
            log_file.flush()

            i += 1

def get_randomized_permutation(X_train, X_validation):
    np.random.seed(int(time.time() * 1000) % 2**32)
    randomized_perm = np.random.permutation(X_train.columns)[: max(
        1, int(np.random.rand() * len(X_train.columns)))]
    not_in_perm = [x for x in X_train.columns if x not in randomized_perm]

    X_train_modified = X_train.drop(not_in_perm, axis=1)
    X_validation_modified = X_validation.drop(not_in_perm, axis=1)

    return randomized_perm, X_train_modified, X_validation_modified
    

def main():
    project = hopsworks.login(project="id2223_pierrelf_emilk2")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="winequality_balanced_typed", version=1)
    query = fg.select_all()
    feature_view = fs.get_or_create_feature_view(name="winequality_balanced_typed",
                                                 version=1,
                                                 description="Read from winequality dataset",
                                                 labels=["quality"],
                                                 query=query)
    feature_view.delete_all_training_datasets()

    num_threads = 1
    if len(sys.argv) > 1:
        num_threads = sys.argv[1]

    for i in range(int(num_threads)):
        X_train, X_validation, X_test, y_train, y_validation, y_test = feature_view.train_validation_test_split(
            0.2, 0.2)
        
        p = Process(target=train, args=(i, X_train, X_validation,
                    X_test, y_train, y_validation, y_test,))
        p.start()


if __name__ == "__main__":
    main()
