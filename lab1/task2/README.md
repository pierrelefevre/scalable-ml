# ID2223 Scalable ML - Lab 1, Task 2 - Wine quality
![Wine tasting group](assets/image.png)
by Emil Karlsson, Pierre Le Fevre

## Task
Predict wine quality of wines from Portugal using the provided dataset.
Dataset is unevenly distributed, there are very few high quality wines.

Could be classification or regression problem.


## Tips from Jim
Might want to use xgboost?

## Some notes from https://www.youtube.com/watch?v=W25TEa93T_I
Lower MSE: Go for regression
Accuracy: Go for classification

## Data
### Run 1
KNN_44, 0.571, [['volatile_acidity', 'residual_sugar', 'chlorides', 'density', 'ph', 'alcohol']]
KNN_39, 0.568, [['volatile_acidity', 'residual_sugar', 'chlorides', 'ph', 'sulphates', 'alcohol']]
KNN_35, 0.565, [['volatile_acidity', 'chlorides', 'ph', 'sulphates', 'alcohol']]
KNN_28, 0.565, [['volatile_acidity', 'chlorides', 'density', 'alcohol', 'type']]
KNN_15, 0.559, [['volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'density', 'ph', 'sulphates', 'alcohol']]
KNN_32, 0.559, [['volatile_acidity', 'alcohol', 'type']]
KNN_48, 0.558, [['volatile_acidity', 'citric_acid', 'density', 'alcohol']]
KNN_35, 0.558, [['fixed_acidity', 'volatile_acidity', 'sulphates', 'alcohol']]
KNN_35, 0.556, [['volatile_acidity', 'chlorides', 'alcohol', 'type']]
KNN_45, 0.554, [['volatile_acidity', 'citric_acid', 'density', 'ph', 'alcohol']]
KNN_46, 0.554, [['fixed_acidity', 'volatile_acidity', 'chlorides', 'density', 'sulphates', 'alcohol']]
KNN_26, 0.554, [['volatile_acidity', 'citric_acid', 'residual_sugar', 'ph', 'sulphates', 'alcohol', 'type']]
KNN_27, 0.554, [['volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'density', 'ph', 'sulphates', 'alcohol']]
KNN_22, 0.552, [['fixed_acidity', 'volatile_acidity', 'citric_acid', 'density', 'ph', 'alcohol']]
KNN_37, 0.550, [['volatile_acidity', 'alcohol', 'type']]
KNN_47, 0.550, [['volatile_acidity', 'citric_acid', 'density', 'sulphates', 'alcohol']]
KNN_32, 0.549, [['volatile_acidity', 'citric_acid', 'density', 'ph', 'alcohol', 'type']]
KNN_30, 0.549, [['volatile_acidity', 'citric_acid', 'chlorides', 'ph', 'alcohol']]
KNN_41, 0.548, [['volatile_acidity', 'chlorides', 'density', 'sulphates', 'alcohol']]
KNN_44, 0.546, [['volatile_acidity', 'chlorides', 'sulphates', 'alcohol']]
KNN_43, 0.546, [['volatile_acidity', 'chlorides', 'free_sulfur_dioxide', 'ph', 'sulphates', 'alcohol']]
KNN_21, 0.546, [['volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'sulphates', 'alcohol', 'type']]
KNN_43, 0.542, [['volatile_acidity', 'alcohol', 'type']]
KNN_43, 0.541, [['fixed_acidity', 'volatile_acidity', 'citric_acid', 'chlorides', 'sulphates', 'alcohol', 'type']]
KNN_19, 0.521, [['volatile_acidity', 'chlorides', 'ph', 'alcohol', 'type']]

### Run 2
RFClassi ({'max_depth': 10, 'n_estimators': 200}) score: 0.601 mse: 0.530 [['volatile_acidity', 'total_sulfur_dioxide', 'fixed_acidity', 'density', 'free_sulfur_dioxide', 'ph', 'chlorides', 'alcohol', 'sulphates', 'type', 'residual_sugar']]
RFClassi ({'max_depth': 10, 'n_estimators': 200}) score: 0.598 mse: 0.548 [['volatile_acidity', 'residual_sugar', 'fixed_acidity', 'density', 'citric_acid', 'alcohol', 'chlorides', 'total_sulfur_dioxide', 'ph', 'free_sulfur_dioxide', 'sulphates']]
RFClassi ({'max_depth': None, 'n_estimators': 200}) score: 0.594 mse: 0.531 [['volatile_acidity', 'density', 'citric_acid', 'sulphates', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'fixed_acidity', 'residual_sugar', 'chlorides', 'alcohol']]
RFClassi ({'max_depth': 20, 'n_estimators': 200}) score: 0.593 mse: 0.538 [['volatile_acidity', 'citric_acid', 'density', 'alcohol', 'chlorides', 'fixed_acidity', 'residual_sugar', 'total_sulfur_dioxide', 'free_sulfur_dioxide', 'ph', 'sulphates']]
RFClassi ({'max_depth': None, 'n_estimators': 200}) score: 0.593 mse: 0.584 [['sulphates', 'ph', 'volatile_acidity', 'total_sulfur_dioxide', 'chlorides', 'residual_sugar', 'citric_acid', 'alcohol', 'type', 'density', 'free_sulfur_dioxide']]
RFClassi ({'max_depth': 10, 'n_estimators': 200}) score: 0.591 mse: 0.580 [['free_sulfur_dioxide', 'residual_sugar', 'alcohol', 'ph', 'fixed_acidity', 'density', 'volatile_acidity', 'chlorides', 'total_sulfur_dioxide', 'type']]
RFClassi ({'max_depth': 20, 'n_estimators': 200}) score: 0.588 mse: 0.570 [['fixed_acidity', 'chlorides', 'free_sulfur_dioxide', 'volatile_acidity', 'total_sulfur_dioxide', 'alcohol', 'sulphates', 'ph', 'citric_acid', 'type']]
RFClassi ({'max_depth': 10, 'n_estimators': 200}) score: 0.588 mse: 0.557 [['chlorides', 'free_sulfur_dioxide', 'ph', 'total_sulfur_dioxide', 'volatile_acidity', 'residual_sugar', 'citric_acid', 'alcohol', 'fixed_acidity', 'sulphates', 'type']]
RFClassi ({'max_depth': 10, 'n_estimators': 200}) score: 0.588 mse: 0.572 [['ph', 'sulphates', 'alcohol', 'citric_acid', 'total_sulfur_dioxide', 'volatile_acidity', 'residual_sugar', 'fixed_acidity', 'density', 'type', 'chlorides']]
RFClassi ({'max_depth': 20, 'n_estimators': 200}) score: 0.587 mse: 0.570 [['ph', 'residual_sugar', 'sulphates', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'fixed_acidity', 'volatile_acidity', 'chlorides', 'alcohol']]
RFClassi ({'max_depth': 10, 'n_estimators': 100}) score: 0.585 mse: 0.570 [['chlorides', 'volatile_acidity', 'citric_acid', 'free_sulfur_dioxide', 'ph', 'alcohol', 'residual_sugar', 'sulphates', 'fixed_acidity', 'total_sulfur_dioxide', 'type']]
RFClassi ({'max_depth': None, 'n_estimators': 200}) score: 0.583 mse: 0.569 [['volatile_acidity', 'sulphates', 'citric_acid', 'free_sulfur_dioxide', 'density', 'alcohol', 'residual_sugar', 'fixed_acidity', 'total_sulfur_dioxide', 'ph', 'chlorides']]
RFClassi ({'max_depth': 10, 'n_estimators': 200}) score: 0.583 mse: 0.572 [['residual_sugar', 'chlorides', 'alcohol', 'density', 'free_sulfur_dioxide', 'volatile_acidity', 'fixed_acidity', 'total_sulfur_dioxide', 'sulphates']]
RFClassi ({'max_depth': None, 'n_estimators': 100}) score: 0.583 mse: 0.571 [['fixed_acidity', 'residual_sugar', 'ph', 'free_sulfur_dioxide', 'type', 'citric_acid', 'chlorides', 'density', 'volatile_acidity', 'alcohol', 'total_sulfur_dioxide']]
RFClassi ({'max_depth': 10, 'n_estimators': 100}) score: 0.583 mse: 0.596 [['alcohol', 'sulphates', 'free_sulfur_dioxide', 'density', 'type', 'fixed_acidity', 'total_sulfur_dioxide', 'residual_sugar', 'ph', 'volatile_acidity']]
RFClassi ({'max_depth': None, 'n_estimators': 200}) score: 0.581 mse: 0.608 [['free_sulfur_dioxide', 'sulphates', 'residual_sugar', 'citric_acid', 'total_sulfur_dioxide', 'type', 'volatile_acidity', 'fixed_acidity', 'alcohol', 'density', 'chlorides']]
RFClassi ({'max_depth': 20, 'n_estimators': 200}) score: 0.581 mse: 0.551 [['chlorides', 'sulphates', 'density', 'fixed_acidity', 'volatile_acidity', 'ph', 'alcohol', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'type', 'citric_acid']]
RFClassi ({'max_depth': 20, 'n_estimators': 200}) score: 0.579 mse: 0.594 [['volatile_acidity', 'ph', 'citric_acid', 'fixed_acidity', 'alcohol', 'density', 'residual_sugar', 'sulphates', 'free_sulfur_dioxide']]
RFClassi ({'max_depth': 10, 'n_estimators': 100}) score: 0.578 mse: 0.581 [['sulphates', 'total_sulfur_dioxide', 'density', 'alcohol', 'free_sulfur_dioxide', 'chlorides', 'residual_sugar', 'fixed_acidity', 'volatile_acidity', 'type', 'ph']]
RFClassi ({'max_depth': 10, 'n_estimators': 200}) score: 0.578 mse: 0.572 [['density', 'type', 'citric_acid', 'sulphates', 'residual_sugar', 'ph', 'total_sulfur_dioxide', 'fixed_acidity', 'volatile_acidity', 'free_sulfur_dioxide']]
RFClassi ({'max_depth': None, 'n_estimators': 200}) score: 0.577 mse: 0.594 [['sulphates', 'alcohol', 'free_sulfur_dioxide', 'citric_acid', 'total_sulfur_dioxide', 'volatile_acidity', 'type', 'chlorides', 'ph', 'residual_sugar', 'fixed_acidity']]
RFClassi ({'max_depth': 10, 'n_estimators': 200}) score: 0.576 mse: 0.588 [['alcohol', 'residual_sugar', 'volatile_acidity', 'sulphates', 'chlorides', 'fixed_acidity', 'total_sulfur_dioxide', 'density', 'free_sulfur_dioxide', 'citric_acid']]
RFClassi ({'max_depth': 10, 'n_estimators': 200}) score: 0.576 mse: 0.578 [['free_sulfur_dioxide', 'citric_acid', 'type', 'sulphates', 'ph', 'total_sulfur_dioxide', 'density', 'volatile_acidity', 'residual_sugar', 'alcohol']]
RFClassi ({'max_depth': None, 'n_estimators': 100}) score: 0.576 mse: 0.598 [['volatile_acidity', 'total_sulfur_dioxide', 'sulphates', 'alcohol', 'residual_sugar', 'density', 'citric_acid', 'ph', 'fixed_acidity', 'type', 'free_sulfur_dioxide']]
RFClassi ({'max_depth': 20, 'n_estimators': 200}) score: 0.575 mse: 0.571 [['free_sulfur_dioxide', 'total_sulfur_dioxide', 'citric_acid', 'chlorides', 'ph', 'sulphates', 'alcohol', 'type', 'fixed_acidity', 'density', 'volatile_acidity']]
RFClassi ({'max_depth': 20, 'n_estimators': 200}) score: 0.574 mse: 0.563 [['volatile_acidity', 'total_sulfur_dioxide', 'fixed_acidity', 'ph', 'citric_acid', 'residual_sugar', 'alcohol', 'sulphates', 'free_sulfur_dioxide', 'type']]
RFClassi ({'max_depth': 20, 'n_estimators': 200}) score: 0.573 mse: 0.589 [['citric_acid', 'fixed_acidity', 'ph', 'chlorides', 'total_sulfur_dioxide', 'alcohol', 'density', 'sulphates', 'residual_sugar', 'volatile_acidity', 'free_sulfur_dioxide']]
RFClassi ({'max_depth': 10, 'n_estimators': 200}) score: 0.573 mse: 0.567 [['chlorides', 'density', 'citric_acid', 'alcohol', 'type', 'free_sulfur_dioxide', 'volatile_acidity', 'residual_sugar']]
RFClassi ({'max_depth': 20, 'n_estimators': 200}) score: 0.572 mse: 0.604 [['residual_sugar', 'total_sulfur_dioxide', 'alcohol', 'ph', 'sulphates', 'volatile_acidity']]
RFClassi ({'max_depth': None, 'n_estimators': 100}) score: 0.572 mse: 0.609 [['sulphates', 'residual_sugar', 'ph', 'density', 'total_sulfur_dioxide', 'free_sulfur_dioxide', 'alcohol', 'volatile_acidity', 'type', 'fixed_acidity']]
RFClassi ({'max_depth': 20, 'n_estimators': 200}) score: 0.572 mse: 0.622 [['volatile_acidity', 'residual_sugar', 'sulphates', 'chlorides', 'ph', 'density', 'citric_acid', 'alcohol', 'free_sulfur_dioxide', 'total_sulfur_dioxide']]
RFClassi ({'max_depth': 10, 'n_estimators': 100}) score: 0.570 mse: 0.585 [['density', 'volatile_acidity', 'free_sulfur_dioxide', 'ph', 'alcohol', 'sulphates', 'fixed_acidity', 'residual_sugar', 'citric_acid', 'type']]
RFClassi ({'max_depth': None, 'n_estimators': 100}) score: 0.570 mse: 0.559 [['residual_sugar', 'ph', 'sulphates', 'citric_acid', 'free_sulfur_dioxide', 'type', 'total_sulfur_dioxide', 'fixed_acidity', 'volatile_acidity', 'alcohol']]
RFClassi ({'max_depth': 20, 'n_estimators': 200}) score: 0.570 mse: 0.585 [['alcohol', 'volatile_acidity', 'sulphates', 'type', 'ph', 'citric_acid', 'total_sulfur_dioxide', 'chlorides', 'fixed_acidity', 'residual_sugar', 'free_sulfur_dioxide']]
RFClassi ({'max_depth': None, 'n_estimators': 200}) score: 0.570 mse: 0.597 [['chlorides', 'residual_sugar', 'fixed_acidity', 'total_sulfur_dioxide', 'free_sulfur_dioxide', 'alcohol', 'type', 'ph', 'sulphates', 'volatile_acidity', 'density']]
RFClassi ({'max_depth': 20, 'n_estimators': 200}) score: 0.570 mse: 0.554 [['residual_sugar', 'total_sulfur_dioxide', 'density', 'type', 'chlorides', 'sulphates', 'volatile_acidity', 'fixed_acidity', 'free_sulfur_dioxide', 'ph', 'citric_acid']]
RFClassi ({'max_depth': None, 'n_estimators': 200}) score: 0.569 mse: 0.595 [['fixed_acidity', 'volatile_acidity', 'total_sulfur_dioxide', 'type', 'alcohol', 'ph', 'chlorides', 'citric_acid', 'free_sulfur_dioxide', 'sulphates']]
RFClassi ({'max_depth': None, 'n_estimators': 200}) score: 0.568 mse: 0.622 [['ph', 'citric_acid', 'alcohol', 'sulphates', 'residual_sugar', 'free_sulfur_dioxide', 'fixed_acidity', 'density', 'type', 'volatile_acidity', 'chlorides']]
RFClassi ({'max_depth': 10, 'n_estimators': 200}) score: 0.568 mse: 0.591 [['free_sulfur_dioxide', 'sulphates', 'fixed_acidity', 'type', 'volatile_acidity', 'ph', 'alcohol', 'residual_sugar', 'citric_acid', 'density', 'chlorides']]
RFClassi ({'max_depth': 20, 'n_estimators': 200}) score: 0.566 mse: 0.586 [['fixed_acidity', 'total_sulfur_dioxide', 'type', 'alcohol', 'sulphates', 'ph', 'chlorides', 'free_sulfur_dioxide', 'volatile_acidity', 'citric_acid']]
RFClassi ({'max_depth': 10, 'n_estimators': 200}) score: 0.563 mse: 0.619 [['volatile_acidity', 'ph', 'type', 'total_sulfur_dioxide', 'density', 'free_sulfur_dioxide', 'chlorides', 'alcohol', 'fixed_acidity', 'sulphates']]
RFClassi ({'max_depth': None, 'n_estimators': 200}) score: 0.562 mse: 0.607 [['alcohol', 'fixed_acidity', 'volatile_acidity', 'citric_acid', 'ph', 'free_sulfur_dioxide', 'residual_sugar', 'density', 'total_sulfur_dioxide', 'sulphates', 'chlorides']]
RFClassi ({'max_depth': None, 'n_estimators': 200}) score: 0.560 mse: 0.619 [['type', 'citric_acid', 'sulphates', 'density', 'alcohol', 'residual_sugar', 'free_sulfur_dioxide', 'volatile_acidity', 'ph', 'total_sulfur_dioxide', 'chlorides']]
RFClassi ({'max_depth': 10, 'n_estimators': 100}) score: 0.559 mse: 0.614 [['residual_sugar', 'volatile_acidity', 'fixed_acidity', 'chlorides', 'total_sulfur_dioxide', 'type', 'density', 'alcohol', 'free_sulfur_dioxide', 'citric_acid', 'ph']]
RFClassi ({'max_depth': 10, 'n_estimators': 100}) score: 0.556 mse: 0.645 [['volatile_acidity', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'alcohol', 'citric_acid']]
