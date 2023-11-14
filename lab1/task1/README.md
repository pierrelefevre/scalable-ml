# ID2223 Scalable ML - Lab 1, Task 1 - Iris prediction
![scientist looking at iris flower](../assets/task1.png)
by Emil Karlsson, Pierre Le Fevre

## Task
Train a classifier on the small iris flower dataset containing the length and width of sepal and petal of iris flowers.

## Backend solution
The proposed solution is a machine learning system in four main parts. 
1. Data preprocessing (iris-eda-and-backfill-feature-group.ipynb)
    - The raw dataset is imported
    - Data is analysed to visualize the data and grok what we will be classifying
    - Dataset it uploaded to Hopsworks

2. Model training (iris-training-pipeline.ipynb)
    - Dataset is fetched
    - KNeighborsClassifier is trained
    - Model is saved to Hopsworks

3. Daily flower generation (iris-feature-pipeline-daily.py)
    - Using some static bounds determined from the prior EDA, we generate a daily flower
    - The flower is saved to Hopsworks

4. Metrics generator (iris-batch-inference-pipeline.py)
    - 

## Frontend
The user accessible part of this project is made possible through Gradio applications hosted on Huggingface Spaces.

1. [Iris](https://huggingface.co/spaces/pierrelf/iris)
    - 
2. [Iris Monitor](https://huggingface.co/spaces/pierrelf/iris-monitor)
    -

## Results
sifushdfiu

## Discussion
While this project was relatively straightforward given most of the code was provided, here are some issues that we encountered.

### Troubles with Modal
From the way Modal presents itself, as a super simple python runner, one would expect it to be trouble free. It does however seem that our scheduled jobs simply do not run again, and get deleted after a few days. In the GUI we can clearly see the scheduled time, however the field "next run" is always "Never".

### Troubles with Hopsworks
Similarly to Modal, Hopsworks presents itself as a plug and play solution integrating with Python. While this is true, the availability has been horrendous, the quotas of the free tier are hit by doing any kind of simple work.

## Conclusion
While the project was fun to do, the tools used were not very reliable. We would not recommend using the free tiers of Modal or Hopsworks for any serious work.