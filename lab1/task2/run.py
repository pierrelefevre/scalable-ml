from sklearn.ensemble import RandomForestRegressor

import hopsworks
import sys


def get_model():
    # parameters from the results of model_finder.py
    # see README.md for more details

    return RandomForestRegressor(n_estimators=200, max_depth=None)


def train(X_train, X_test, y_train, y_test):
    model = get_model()
    model.fit(X_train, y_train)

    return model

def main():
    project = hopsworks.login(project="id2223_pierrelf_emilk2")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="winequality_typed_balanced", version=1)
    query = fg.select_all()
    feature_view = fs.get_or_create_feature_view(name="winequality_typed_balanced",
                                                 version=1,
                                                 description="Read from winequality dataset",
                                                 labels=["quality"],
                                                 query=query)
    feature_view.delete_all_training_datasets()
    X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)
    

    model = train(X_train, X_test, y_train, y_test)


    score = model.score(X_test, y_test)
    print(f"Score: {score}")


if __name__ == "__main__":
    main()
