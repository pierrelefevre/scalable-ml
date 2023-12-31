from PIL import Image
from datetime import datetime
from sklearn.metrics import confusion_matrix

import pandas as pd
import dataframe_image as dfi
import seaborn as sns

import hopsworks
import joblib
import shutil
import requests


def load_model(project):
    mr = project.get_model_registry()
    model = mr.get_model("wine_model")
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")

    return model


def save_wine_image(quality, path):
    url = f"https://raw.githubusercontent.com/pierrelefevre/scalable-ml/main/lab1/task2/img/{quality}.png"

    img = Image.open(requests.get(url, stream=True).raw)            
    img.save(path)


def upload_confusion_matrix(dataset_api, actuals, predictions):
    all_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    results = confusion_matrix(actuals, predictions, labels=all_classes)

    df_cm = pd.DataFrame(results, index=all_classes, columns=all_classes)

    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()
    fig.savefig("./img/confusion_matrix_wine.png")
    dataset_api.upload("./img/confusion_matrix_wine.png", "Resources/images", overwrite=True)


def upload_prediction(dataset_api, pred_fg, predicted_quality, actual_quality):
    
    # write prediction to feature store
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [predicted_quality],
        'label': [actual_quality],
        'datetime': [now],
       }
    pred_df = pd.DataFrame(data)

    # fetch the history to be able to fetch the latest predictions
    history_df = pred_fg.read()
    history_df = pd.concat([history_df, pred_df])

    # fill with empty 

    # insert afterwards to ensure that we don't insert pred_df twice
    pred_fg.insert(pred_df, write_options={"wait_for_job" : False})

    upload_confusion_matrix(dataset_api, history_df["label"], history_df["prediction"])

    # write image to hopsworks for visualization
    df_recent = history_df.tail(4)
    dfi.export(df_recent, './img/df_recent_wine.png', table_conversion = 'matplotlib')
    dataset_api.upload("./img/df_recent_wine.png", "Resources/images", overwrite=True)
    
    save_wine_image(predicted_quality, "./img/latest_wine.png")
    dataset_api.upload("./img/latest_wine.png", "Resources/images", overwrite=True)

    save_wine_image(actual_quality, "./img/actual_wine.png")
    dataset_api.upload("./img/actual_wine.png", "Resources/images", overwrite=True)



def main():
    name = "winequality_typed_balanced"

    project = hopsworks.login(project="id2223_pierrelf_emilk2")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=name)
    fv = fs.get_feature_view(name=name)
    pred_fg = fs.get_or_create_feature_group(
        name=f"{name}_predictions",
        version=1,
        primary_key=["datetime"],
        description="Wine quality predictions",
    )

    dataset_api = project.get_dataset_api()

    print("Reading latest wine quality...")
    batch_data = fv.get_batch_data()

    print("Loading model...")
    model = load_model(project)

    y_pred = model.predict(batch_data)
    predicted_quality = int(round(y_pred[y_pred.size-1]))
    print("Predicted quality: {}".format(predicted_quality))

    actual_quality = int(round(fg.read().iloc[-1]["quality"]))
    print("Actual quality: {}".format(actual_quality))

    upload_prediction(dataset_api, pred_fg, predicted_quality, actual_quality)
    
    

if __name__ == '__main__':
    main()