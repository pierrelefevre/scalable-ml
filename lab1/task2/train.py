#
# train.py - gets dataset, trains model and uploads to Hopsworks
# -- run once --
#

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from hsml.schema import Schema
from hsml.model import ModelSchema

import seaborn as sns
import pandas as pd

import hopsworks
import sys
import os
import joblib

def get_model():
    # parameters from the results of model_finder.py
    # see README.md for more details

    return RandomForestRegressor(n_estimators=200, max_depth=None)

def cf_heatmap(results):
    df_cm = pd.DataFrame(results, ['True Setosa', 'True Versicolor', 'True Virginica'],
                     ['Pred Setosa', 'Pred Versicolor', 'Pred Virginica'])
    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()

    return fig

def get_model_schema(X_train, y_train):
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    return model_schema

def save_model(project, model, model_schema, metrics, results):
    fig = cf_heatmap(results)

    # We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
    mr = project.get_model_registry()

    # The contents of the 'iris_model' directory will be saved to the model registry. Create the dir, first.
    model_dir="wine_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)

    # Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
    joblib.dump(model, model_dir + "/wine_model.pkl")
    fig.savefig(model_dir + "/confusion_matrix.png")
    

    # Create an entry in the model registry that includes the model's name, desc, metrics
    iris_model = mr.python.create_model(
        name="wine_model", 
        metrics={"accuracy" : metrics['accuracy']},
        model_schema=model_schema,
        description="Wine quality prediction model",
    )

    # Upload the model to the model registry, including all files in 'model_dir'
    iris_model.save(model_dir)

def train(X_train, X_test, y_train, y_test):
    model = get_model()
    model.fit(X_train, y_train)

    # Evaluate model performance using the features from the test set (X_test)
    y_pred = model.predict(X_test)

    metrics = classification_report(y_test, y_pred, output_dict=True)
    results = confusion_matrix(y_test, y_pred)

    return model, metrics, results

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
    

    model, metrics, results = train(X_train, X_test, y_train, y_test)


    score = model.score(X_test, y_test)
    print(f"Model score: {score}")


    print("Saving model...")
    save_model(project, model, get_model_schema(X_train, y_train), metrics, results)


if __name__ == "__main__":
    main()
