from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import hopsworks

# generate a random entry in the wine table, based on the dataset's distribution
def generate_wine(dataframe):
    sampled_row = []
    for column in dataframe.columns:
        # skip quality column since it will be predicted by KNN later
        if column == 'quality':
            continue
        
        sampled_value = dataframe[column].sample(1).values[0]
        sampled_row.append(sampled_value)

    return sampled_row
    


# return the closest score to the given row (closest in terms of features)
def get_closest_score(df, row):
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=3) # You can change the number of neighbors
    knn.fit(X_train_scaled, y_train)

    # Predict the closest quality for a new row of features
    # Replace this with your specific row
    new_row_scaled = scaler.transform([row])
    predicted_quality = knn.predict(new_row_scaled)

    return predicted_quality[0]


def main():
    fg = hopsworks.login(project="id2223_pierrelf_emilk2").get_feature_store().get_feature_group(name="winequality_typed_balanced", version=1)
    df = fg.read()

    random_wine = generate_wine(df)
    predicted_quality = get_closest_score(df, random_wine)

    print(f"Generated wine: {random_wine} with predicted quality: {predicted_quality}")

    # write to prediction feature 

    


if __name__ == '__main__':
    main()