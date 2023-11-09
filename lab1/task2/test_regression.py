
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import hopsworks

project = hopsworks.login(project="id2223_pierrelf_emilk2")
fs = project.get_feature_store()
fg = fs.get_feature_group(name="winequality", version=1)
query = fg.select_all()
feature_view = fs.get_or_create_feature_view(name="winequality",
                                                version=1,
                                                description="Read from winequality dataset",
                                                labels=["quality"],
                                                query=query)

X_train, X_validation, X_test, y_train, y_validation, y_test = feature_view.train_validation_test_split(
            0.2, 0.2)

# List of models
models = [
    ("Linear Regression", LinearRegression()),
    ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("Support Vector Machine", SVR())
]

def evaluate_model(name, model, X_train, y_train, X_validation, y_validation):
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_validation)
    mse = mean_squared_error(y_validation, y_pred)
    print(f"{name}: Validation Mean Squared Error = {mse:.2f}")

# Evaluate each model
for name, model in models:
    evaluate_model(name, model, X_train, y_train, X_validation, y_validation)