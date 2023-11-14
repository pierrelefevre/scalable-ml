import hopsworks
import joblib
import shutil

def load_model(project):
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")

def save_wine_image(quality, path):
    shutil.copyfile(f"./img/{quality}.png", path)

def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="iris", version=1)
    fv = fs.get_feature_view(name="iris", version=1)
    dataset_api = project.get_dataset_api()

    batch_data = fv.get_batch_data()
    model = load_model(project)

    y_pred = model.predict(batch_data)
    predicted_quality = y_pred[0]
    print("Predicted quality: {}".format(predicted_quality))

    df = fg.read() 
    actual_quality = df.iloc[-1]["quality"]
    print("Actual quality: {}".format(actual_quality))

    # write image to hopsworks
    save_wine_image(predicted_quality, "./img/latest_wine.png")
    save_wine_image(actual_quality, "./img/actual_wine.png")

    dataset_api.upload("./img/latest_wine.png", "Resources/images", overwrite=True)
    dataset_api.upload("./img/actual_wine.png", "Resources/images", overwrite=True)
    
    

if __name__ == '__main__':
    main()