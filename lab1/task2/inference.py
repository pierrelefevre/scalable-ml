from PIL import Image

import pandas as pd
import dataframe_image as dfi

import hopsworks
import joblib
import shutil
import requests
import datetime

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

def upload_prediction(pred_fg, predicted_quality, actual_quality):
    
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

    # insert afterwards to ensure that we don't insert pred_df twice
    pred_fg.insert(pred_df, write_options={"wait_for_job" : False})


    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)

def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="winequality_typed_balanced")
    fv = fs.get_feature_view(name="winequality_typed_balanced")
    fg_pred = fs.get_or_create_feature_group(name="winequality_typed_balanced_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Predictions for winequality_typed_balanced",
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

    # write image to hopsworks
    save_wine_image(predicted_quality, "./img/latest_wine.png")
    save_wine_image(actual_quality, "./img/actual_wine.png")

    dataset_api.upload("./img/latest_wine.png", "Resources/images", overwrite=True)
    dataset_api.upload("./img/actual_wine.png", "Resources/images", overwrite=True)


    
    

if __name__ == '__main__':
    main()