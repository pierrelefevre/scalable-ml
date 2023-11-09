import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
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

# # List of models
# models = [
#     ("LinReg  ", LinearRegression()),
#     ("RForest ", RandomForestRegressor(n_estimators=100, random_state=42)),
#     ("SVR     ", SVR()),
#     ("DTreeReg", DecisionTreeRegressor(max_depth=10))
# ]

models = [
    ("Linear Regression", LinearRegression(), {}),
    ("Linear Regression", Pipeline([
        ("poly_features", PolynomialFeatures()),
        ("linear_regression", LinearRegression())
    ]), {'poly_features__degree': [1, 2, 3]}),
    ("Random Forest", RandomForestRegressor(), {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}),
    ("Support Vector Machine", SVR(), {'C': [1, 10], 'gamma': ['scale', 'auto']}),
    ("Decision Tree", DecisionTreeRegressor(), {'max_depth': [None, 10, 20], 'min_samples_split': [2, 10]}),
    ("Gradient Boosting", GradientBoostingRegressor(), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]})
]

# def evaluate_model(name, model, X_train, y_train, X_validation, y_validation):
#     model.fit(X_train, y_train.values.ravel())
#     y_pred = model.predict(X_validation)
#     score = model.score(X_validation, y_validation)
#     mse = mean_squared_error(y_validation, y_pred)
#     print(f"{name}: \t Validation Score: {score:.2f}, \tMSE: {mse:.2f}")

def evaluate_model(name, model, params, X_train, y_train, X_validation, y_validation):
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train.values.ravel())
    best_model = grid_search.best_estimator_
    score = model.score(X_validation, y_validation)
    y_pred = best_model.predict(X_validation)
    mse = mean_squared_error(y_validation, y_pred)
    print(f"--- {name} ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Validation Score: {score:.2f}, \tMSE: {mse:.2f}")

# Evaluate each model
for name, model, params in models:
    evaluate_model(name, params, model, X_train, y_train, X_validation, y_validation)