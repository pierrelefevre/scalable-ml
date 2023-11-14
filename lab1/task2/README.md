# ID2223 Scalable ML - Lab 1, Task 2 - Wine quality
![Wine tasting group](../assets/task2.png)
by Emil Karlsson, Pierre Le Fevre

## Task
Predict wine quality of wines from Portugal using the provided dataset.
Dataset is unevenly distributed, there are very few high quality wines.

Could be classification or regression problem.
## Proposed solution
### Backend 
The proposed solution is a machine learning system in five main parts. 
1. Data preprocessing (initialize.ipynb)
    - The raw dataset is imported
    - Data is analysed to visualize the data and grok what we will be classifying
    - Data is cleaned and balanced
    - Dataset it uploaded to Hopsworks

2. Model finder (model_finder.py)
    - Dataset is fetched
    - A number of models are trained
    - Highest scoring model is parsed

3. Model training (train.ipynb)
    - Using the best model from the model finder, we train it on the full dataset
    - The model is saved to Hopsworks

4. Daily wine generation (daily.py)
    - Generate a random wine by 
        - Randomizing the values
        - Finding the nearest neighbor
        - Skewing the values a bit
        - Predicting the quality
    - The wine is saved to Hopsworks

5. Metrics generator (inference.py)
    - ???

### Frontend
The user accessible part of this project is made possible through Gradio applications hosted on Huggingface Spaces.

1. [Wine](https://huggingface.co/spaces/pierrelf/wine)
    - ???
2. [Wine Monitor](https://huggingface.co/spaces/pierrelf/wine-monitor)
    - ???



## Results
???

## Discussion
### Troubles with Hopworks
Similarly to the troubles encountered in task 1, the spotty availability of Hopsworks has not exactly been helpful in our workflow. 
We also noticed that the train_test_validation_split function saves the split dataset on Hopsworks, but has a limit of 100 saved datasets. This meant we had to delete the old datasets to be able to run the notebook again, something which could be avoided by simply saving the split dataset locally.

## Conclusion
???

## Notes
### Tips from Jim
- Might want to use xgboost?
- Generate wines by randomizing, find nearest neighbor, skew it a bit and then predict

### Some notes from https://www.youtube.com/watch?v=W25TEa93T_I
Lower MSE: Go for regression
Accuracy: Go for classification

## Model finder run outputs
### Run 4 (Balanced data, added back types)
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.942 mse: 0.250 [['sulphates', 'chlorides', 'fixed_acidity', 'ph', 'alcohol', 'volatile_acidity', 'total_sulfur_dioxide', 'residual_sugar', 'citric_acid']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.940 mse: 0.256 [['alcohol', 'residual_sugar', 'chlorides', 'fixed_acidity', 'sulphates', 'ph', 'volatile_acidity', 'citric_acid', 'total_sulfur_dioxide']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.939 mse: 0.266 [['alcohol', 'chlorides', 'ph', 'residual_sugar', 'sulphates', 'citric_acid', 'fixed_acidity', 'total_sulfur_dioxide', 'volatile_acidity']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.938 mse: 0.269 [['sulphates', 'fixed_acidity', 'total_sulfur_dioxide', 'citric_acid', 'ph', 'alcohol', 'volatile_acidity', 'residual_sugar', 'chlorides']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.938 mse: 0.269 [['ph', 'citric_acid', 'volatile_acidity', 'chlorides', 'fixed_acidity', 'total_sulfur_dioxide', 'sulphates', 'alcohol', 'residual_sugar']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.938 mse: 0.269 [['citric_acid', 'volatile_acidity', 'residual_sugar', 'sulphates', 'chlorides', 'alcohol', 'fixed_acidity', 'total_sulfur_dioxide', 'ph']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.937 mse: 0.273 [['residual_sugar', 'ph', 'volatile_acidity', 'chlorides', 'sulphates', 'total_sulfur_dioxide', 'citric_acid', 'type', 'alcohol']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.936 mse: 0.271 [['total_sulfur_dioxide', 'alcohol', 'volatile_acidity', 'residual_sugar', 'type', 'sulphates', 'citric_acid', 'ph', 'fixed_acidity']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.936 mse: 0.278 [['sulphates', 'ph', 'residual_sugar', 'chlorides', 'total_sulfur_dioxide', 'alcohol', 'volatile_acidity', 'fixed_acidity', 'citric_acid']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.936 mse: 0.272 [['alcohol', 'residual_sugar', 'ph', 'sulphates', 'chlorides', 'total_sulfur_dioxide', 'volatile_acidity', 'fixed_acidity', 'citric_acid']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.936 mse: 0.270 [['type', 'citric_acid', 'sulphates', 'residual_sugar', 'fixed_acidity', 'ph', 'total_sulfur_dioxide', 'volatile_acidity', 'alcohol']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.936 mse: 0.275 [['ph', 'citric_acid', 'residual_sugar', 'fixed_acidity', 'volatile_acidity', 'alcohol', 'total_sulfur_dioxide', 'chlorides', 'sulphates']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.936 mse: 0.273 [['volatile_acidity', 'residual_sugar', 'alcohol', 'sulphates', 'total_sulfur_dioxide', 'fixed_acidity', 'ph', 'chlorides', 'citric_acid']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.936 mse: 0.276 [['volatile_acidity', 'citric_acid', 'alcohol', 'sulphates', 'ph', 'chlorides', 'total_sulfur_dioxide', 'residual_sugar']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.936 mse: 0.275 [['residual_sugar', 'volatile_acidity', 'chlorides', 'sulphates', 'alcohol', 'ph', 'citric_acid', 'type', 'total_sulfur_dioxide']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.936 mse: 0.271 [['citric_acid', 'ph', 'volatile_acidity', 'residual_sugar', 'fixed_acidity', 'alcohol', 'total_sulfur_dioxide', 'sulphates', 'chlorides']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.936 mse: 0.272 [['ph', 'citric_acid', 'alcohol', 'sulphates', 'chlorides', 'fixed_acidity', 'volatile_acidity', 'total_sulfur_dioxide', 'residual_sugar']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.936 mse: 0.272 [['residual_sugar', 'volatile_acidity', 'fixed_acidity', 'total_sulfur_dioxide', 'alcohol', 'sulphates', 'ph', 'chlorides', 'type']]
RFRegres ({'max_depth': 20, 'n_estimators': 200}) score: 0.935 mse: 0.277 [['total_sulfur_dioxide', 'sulphates', 'type', 'residual_sugar', 'fixed_acidity', 'citric_acid', 'volatile_acidity', 'alcohol', 'ph']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.935 mse: 0.279 [['ph', 'total_sulfur_dioxide', 'citric_acid', 'residual_sugar', 'chlorides', 'sulphates', 'alcohol', 'volatile_acidity']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.935 mse: 0.275 [['citric_acid', 'volatile_acidity', 'sulphates', 'residual_sugar', 'alcohol', 'fixed_acidity', 'type', 'ph', 'total_sulfur_dioxide']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.935 mse: 0.276 [['alcohol', 'citric_acid', 'fixed_acidity', 'volatile_acidity', 'ph', 'total_sulfur_dioxide', 'chlorides', 'residual_sugar', 'sulphates']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.935 mse: 0.277 [['citric_acid', 'residual_sugar', 'volatile_acidity', 'sulphates', 'ph', 'type', 'chlorides', 'alcohol', 'total_sulfur_dioxide']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.935 mse: 0.274 [['residual_sugar', 'chlorides', 'citric_acid', 'total_sulfur_dioxide', 'ph', 'volatile_acidity', 'alcohol', 'sulphates', 'fixed_acidity']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.935 mse: 0.279 [['volatile_acidity', 'citric_acid', 'residual_sugar', 'alcohol', 'ph', 'chlorides', 'type', 'sulphates', 'total_sulfur_dioxide']]
RFRegres ({'max_depth': 20, 'n_estimators': 200}) score: 0.934 mse: 0.282 [['citric_acid', 'chlorides', 'total_sulfur_dioxide', 'residual_sugar', 'sulphates', 'volatile_acidity', 'fixed_acidity', 'alcohol', 'ph']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.934 mse: 0.284 [['total_sulfur_dioxide', 'citric_acid', 'residual_sugar', 'volatile_acidity', 'ph', 'sulphates', 'alcohol', 'chlorides']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.934 mse: 0.278 [['alcohol', 'ph', 'chlorides', 'residual_sugar', 'citric_acid', 'fixed_acidity', 'total_sulfur_dioxide', 'sulphates', 'volatile_acidity']]
RFRegres ({'max_depth': 20, 'n_estimators': 200}) score: 0.934 mse: 0.281 [['sulphates', 'alcohol', 'fixed_acidity', 'residual_sugar', 'citric_acid', 'volatile_acidity', 'total_sulfur_dioxide', 'chlorides', 'ph']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.933 mse: 0.285 [['citric_acid', 'total_sulfur_dioxide', 'type', 'sulphates', 'fixed_acidity', 'residual_sugar', 'alcohol', 'volatile_acidity', 'ph']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.933 mse: 0.288 [['chlorides', 'citric_acid', 'total_sulfur_dioxide', 'alcohol', 'ph', 'fixed_acidity', 'residual_sugar', 'sulphates', 'volatile_acidity']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.933 mse: 0.279 [['volatile_acidity', 'alcohol', 'type', 'fixed_acidity', 'ph', 'citric_acid', 'total_sulfur_dioxide', 'sulphates', 'residual_sugar']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.933 mse: 0.282 [['volatile_acidity', 'fixed_acidity', 'citric_acid', 'alcohol', 'ph', 'residual_sugar', 'sulphates', 'chlorides', 'total_sulfur_dioxide']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.933 mse: 0.290 [['chlorides', 'total_sulfur_dioxide', 'alcohol', 'residual_sugar', 'fixed_acidity', 'volatile_acidity', 'citric_acid', 'sulphates', 'ph']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.932 mse: 0.289 [['sulphates', 'chlorides', 'ph', 'residual_sugar', 'type', 'alcohol', 'citric_acid', 'total_sulfur_dioxide', 'volatile_acidity']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.932 mse: 0.291 [['total_sulfur_dioxide', 'alcohol', 'fixed_acidity', 'type', 'ph', 'volatile_acidity', 'citric_acid', 'sulphates', 'residual_sugar']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.932 mse: 0.284 [['residual_sugar', 'sulphates', 'citric_acid', 'chlorides', 'volatile_acidity', 'ph', 'alcohol', 'type', 'total_sulfur_dioxide']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.932 mse: 0.290 [['volatile_acidity', 'ph', 'residual_sugar', 'alcohol', 'chlorides', 'total_sulfur_dioxide', 'type', 'citric_acid', 'sulphates']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.932 mse: 0.285 [['alcohol', 'volatile_acidity', 'sulphates', 'fixed_acidity', 'chlorides', 'total_sulfur_dioxide', 'citric_acid', 'residual_sugar', 'ph']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.931 mse: 0.291 [['total_sulfur_dioxide', 'residual_sugar', 'alcohol', 'sulphates', 'fixed_acidity', 'volatile_acidity', 'ph', 'type', 'citric_acid']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.931 mse: 0.295 [['chlorides', 'citric_acid', 'ph', 'sulphates', 'residual_sugar', 'volatile_acidity', 'total_sulfur_dioxide', 'alcohol', 'fixed_acidity']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.931 mse: 0.289 [['alcohol', 'citric_acid', 'volatile_acidity', 'fixed_acidity', 'ph', 'sulphates', 'total_sulfur_dioxide', 'chlorides', 'residual_sugar']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.931 mse: 0.288 [['fixed_acidity', 'sulphates', 'volatile_acidity', 'alcohol', 'ph', 'citric_acid', 'chlorides', 'total_sulfur_dioxide', 'residual_sugar']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.930 mse: 0.301 [['residual_sugar', 'type', 'chlorides', 'alcohol', 'volatile_acidity', 'sulphates', 'total_sulfur_dioxide', 'citric_acid', 'ph']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.930 mse: 0.300 [['residual_sugar', 'type', 'total_sulfur_dioxide', 'alcohol', 'sulphates', 'volatile_acidity', 'fixed_acidity', 'citric_acid', 'ph']]


### Run 3 (Balanced data, dropped some columns)
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.939 mse: 0.261 [['volatile_acidity', 'total_sulfur_dioxide', 'sulphates', 'residual_sugar', 'chlorides', 'alcohol', 'citric_acid', 'ph']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.938 mse: 0.270 [['alcohol', 'ph', 'volatile_acidity', 'fixed_acidity', 'total_sulfur_dioxide', 'chlorides', 'sulphates', 'citric_acid']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.938 mse: 0.270 [['alcohol', 'volatile_acidity', 'chlorides', 'ph', 'residual_sugar', 'citric_acid', 'total_sulfur_dioxide', 'sulphates']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.937 mse: 0.267 [['alcohol', 'sulphates', 'volatile_acidity', 'ph', 'residual_sugar', 'total_sulfur_dioxide', 'chlorides', 'fixed_acidity']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.936 mse: 0.272 [['total_sulfur_dioxide', 'volatile_acidity', 'citric_acid', 'alcohol', 'sulphates', 'ph', 'chlorides', 'residual_sugar']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.936 mse: 0.278 [['total_sulfur_dioxide', 'alcohol', 'volatile_acidity', 'sulphates', 'ph', 'chlorides', 'residual_sugar', 'citric_acid']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.934 mse: 0.281 [['chlorides', 'ph', 'residual_sugar', 'sulphates', 'fixed_acidity', 'alcohol', 'volatile_acidity', 'total_sulfur_dioxide']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.934 mse: 0.284 [['residual_sugar', 'total_sulfur_dioxide', 'citric_acid', 'alcohol', 'ph', 'sulphates', 'volatile_acidity', 'chlorides']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.934 mse: 0.283 [['alcohol', 'total_sulfur_dioxide', 'residual_sugar', 'volatile_acidity', 'fixed_acidity', 'chlorides', 'sulphates', 'citric_acid']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.934 mse: 0.285 [['residual_sugar', 'alcohol', 'fixed_acidity', 'total_sulfur_dioxide', 'volatile_acidity', 'sulphates', 'chlorides', 'ph']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.933 mse: 0.285 [['total_sulfur_dioxide', 'residual_sugar', 'citric_acid', 'chlorides', 'sulphates', 'fixed_acidity', 'alcohol', 'ph']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.933 mse: 0.289 [['alcohol', 'volatile_acidity', 'sulphates', 'ph', 'fixed_acidity', 'citric_acid', 'residual_sugar', 'total_sulfur_dioxide']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.932 mse: 0.298 [['volatile_acidity', 'alcohol', 'fixed_acidity', 'ph', 'citric_acid', 'total_sulfur_dioxide', 'sulphates', 'chlorides']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.932 mse: 0.293 [['alcohol', 'sulphates', 'total_sulfur_dioxide', 'chlorides', 'residual_sugar', 'volatile_acidity', 'fixed_acidity', 'ph']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.931 mse: 0.292 [['volatile_acidity', 'citric_acid', 'residual_sugar', 'total_sulfur_dioxide', 'alcohol', 'sulphates', 'ph', 'chlorides']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.931 mse: 0.298 [['chlorides', 'citric_acid', 'total_sulfur_dioxide', 'residual_sugar', 'fixed_acidity', 'sulphates', 'alcohol', 'volatile_acidity']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.931 mse: 0.291 [['volatile_acidity', 'total_sulfur_dioxide', 'sulphates', 'citric_acid', 'alcohol', 'fixed_acidity', 'chlorides', 'ph']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.931 mse: 0.299 [['ph', 'chlorides', 'volatile_acidity', 'alcohol', 'total_sulfur_dioxide', 'fixed_acidity', 'citric_acid', 'sulphates']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.930 mse: 0.304 [['residual_sugar', 'sulphates', 'citric_acid', 'alcohol', 'total_sulfur_dioxide', 'fixed_acidity', 'chlorides']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.930 mse: 0.296 [['fixed_acidity', 'alcohol', 'chlorides', 'sulphates', 'total_sulfur_dioxide', 'residual_sugar', 'citric_acid', 'ph']]
RFRegres ({'max_depth': None, 'n_estimators': 100}) score: 0.930 mse: 0.304 [['total_sulfur_dioxide', 'sulphates', 'residual_sugar', 'volatile_acidity', 'chlorides', 'alcohol']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.929 mse: 0.302 [['residual_sugar', 'volatile_acidity', 'total_sulfur_dioxide', 'sulphates', 'alcohol', 'citric_acid', 'chlorides', 'fixed_acidity']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.929 mse: 0.298 [['total_sulfur_dioxide', 'ph', 'residual_sugar', 'chlorides', 'volatile_acidity', 'citric_acid', 'alcohol', 'sulphates']]
RFRegres ({'max_depth': 20, 'n_estimators': 200}) score: 0.929 mse: 0.303 [['total_sulfur_dioxide', 'residual_sugar', 'fixed_acidity', 'citric_acid', 'sulphates', 'alcohol', 'ph', 'volatile_acidity']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.929 mse: 0.301 [['sulphates', 'ph', 'fixed_acidity', 'volatile_acidity', 'total_sulfur_dioxide', 'chlorides', 'citric_acid', 'alcohol']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.928 mse: 0.307 [['sulphates', 'chlorides', 'alcohol', 'citric_acid', 'ph', 'residual_sugar', 'total_sulfur_dioxide']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.928 mse: 0.302 [['residual_sugar', 'fixed_acidity', 'citric_acid', 'ph', 'chlorides', 'total_sulfur_dioxide', 'alcohol', 'sulphates']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.928 mse: 0.310 [['ph', 'citric_acid', 'alcohol', 'chlorides', 'total_sulfur_dioxide', 'residual_sugar', 'fixed_acidity', 'volatile_acidity']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.926 mse: 0.313 [['residual_sugar', 'total_sulfur_dioxide', 'sulphates', 'chlorides', 'ph', 'fixed_acidity', 'citric_acid', 'alcohol']]
RFRegres ({'max_depth': 20, 'n_estimators': 200}) score: 0.925 mse: 0.320 [['sulphates', 'ph', 'fixed_acidity', 'chlorides', 'citric_acid', 'residual_sugar', 'alcohol', 'total_sulfur_dioxide']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.925 mse: 0.317 [['residual_sugar', 'total_sulfur_dioxide', 'alcohol', 'ph', 'citric_acid', 'sulphates']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.924 mse: 0.319 [['chlorides', 'total_sulfur_dioxide', 'citric_acid', 'volatile_acidity', 'fixed_acidity', 'alcohol', 'residual_sugar']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.922 mse: 0.332 [['alcohol', 'total_sulfur_dioxide', 'citric_acid', 'sulphates', 'fixed_acidity', 'chlorides', 'volatile_acidity']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.920 mse: 0.339 [['alcohol', 'fixed_acidity', 'volatile_acidity', 'total_sulfur_dioxide', 'citric_acid', 'ph', 'residual_sugar', 'sulphates']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.920 mse: 0.341 [['chlorides', 'volatile_acidity', 'citric_acid', 'alcohol', 'total_sulfur_dioxide', 'fixed_acidity']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.919 mse: 0.344 [['citric_acid', 'total_sulfur_dioxide', 'chlorides', 'alcohol', 'ph', 'sulphates']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.918 mse: 0.340 [['alcohol', 'total_sulfur_dioxide', 'chlorides', 'fixed_acidity', 'volatile_acidity', 'sulphates', 'residual_sugar']]
RFRegres ({'max_depth': 20, 'n_estimators': 200}) score: 0.918 mse: 0.349 [['total_sulfur_dioxide', 'ph', 'volatile_acidity', 'citric_acid', 'alcohol', 'residual_sugar']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.918 mse: 0.351 [['fixed_acidity', 'alcohol', 'volatile_acidity', 'sulphates', 'residual_sugar', 'total_sulfur_dioxide']]
RFRegres ({'max_depth': 20, 'n_estimators': 200}) score: 0.915 mse: 0.364 [['alcohol', 'total_sulfur_dioxide', 'citric_acid', 'volatile_acidity', 'ph', 'residual_sugar']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.915 mse: 0.358 [['residual_sugar', 'alcohol', 'fixed_acidity', 'volatile_acidity', 'citric_acid', 'total_sulfur_dioxide']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.912 mse: 0.374 [['citric_acid', 'volatile_acidity', 'alcohol', 'chlorides', 'residual_sugar', 'fixed_acidity', 'sulphates']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.912 mse: 0.379 [['citric_acid', 'alcohol', 'volatile_acidity', 'residual_sugar', 'fixed_acidity', 'chlorides', 'ph', 'sulphates']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.905 mse: 0.399 [['alcohol', 'ph', 'fixed_acidity', 'volatile_acidity', 'total_sulfur_dioxide']]
RFRegres ({'max_depth': None, 'n_estimators': 200}) score: 0.901 mse: 0.424 [['total_sulfur_dioxide', 'chlorides', 'volatile_acidity', 'fixed_acidity', 'citric_acid', 'ph', 'residual_sugar']]

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