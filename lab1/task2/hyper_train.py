import hopsworks
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_feature_group(name="winequality", version=1)
query = fg.select_all()
feature_view = fs.get_or_create_feature_view(name="winequality",
                                  version=1,
                                  description="Read from winequality dataset",
                                  labels=["quality"],
                                  query=query)


# You can read training data, randomly split into train/test sets of features (X) and labels (y)        
X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)

# Train our model with the Scikit-learn K-nearest-neighbors algorithm using our features (X_train) and labels (y_train)
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, y_train.values.ravel())

# Evaluate model performance using the features from the test set (X_test)
y_pred = model.predict(X_test)

metrics = classification_report(y_test, y_pred, output_dict=True)

print(metrics)