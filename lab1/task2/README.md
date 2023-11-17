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
    - The wine is saved to Hopsworks

5. Metrics generator (inference.py)
    - Predicts the daily wine
    - Calculates the confusion matrix

### Frontend
The user accessible part of this project is made possible through Gradio applications hosted on Huggingface Spaces.

1. [Wine Predictor](https://huggingface.co/spaces/pierrelf/wine)
    - The wine predictor allows testing the model with custom inputs.
2. [Wine Monitor](https://huggingface.co/spaces/pierrelf/wine-monitor)
    - The wine monitor shows the latest generated daily wines and a confusion matrix.



## Results
The wine predictor has a 0.93 accuracy on the test set. The confusion matrix shows it is mostly skewed towards predicting 5s and 6s, which is not surprising given the uneven distribution of the dataset.

## Discussion
### Troubles with Hopworks
Similarly to the troubles encountered in task 1, the spotty availability of Hopsworks has not exactly been helpful in our workflow. 
We also noticed that the train_test_validation_split function saves the split dataset on Hopsworks, but has a limit of 100 saved datasets. This meant we had to delete the old datasets to be able to run the notebook again, something which could be avoided by simply saving the split dataset locally.

### Troubles with Modal
When running cron jobs, Modals seems to inexplicably remove the apps overnight. We decided early on to use GitHub Actions instead, which has been working flawlessly.

## Conclusion
The proposed solution is a machine learning system in five main parts.
The model shows limited utility but a good tech demo for setting up a machine learning system.

## Further work
We would like to retrain the model on the streaming data, however since the data is synthetic this would most likely not improve the model.
We would also like to add more metrics to the model finder.