import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm
import pickle

# Read in data
data_frame = pd.read_csv("breast-cancer-wisconsin//breast-cancer-wisconsin.data.txt")

data_frame.replace("?", -99999, inplace=True)  # Turn Missing Data into severe outliers
data_frame.drop(["id"], axis=1, inplace=True)  # Drop the id column from data_frame. Do it or DIE

X = np.array(data_frame.drop(["class"], axis=1))  # Use all columns as features other than class
y = np.array(data_frame["class"])  # Use the class as the label

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)  # Split the training and test data sets

# Create the classifier model. Support Vector Classifier
clf = svm.SVC()
clf.fit(X_train, y_train)  # Train the model

# open up a path for your pickle file (serialized model)
with open("supportvectormachine.pickle", 'wb') as f:
    # Dump your model into the file
    pickle.dump(clf, f)
# Reopen your pickle model
pickle_in = open("supportvectormachine.pickle", 'rb')
# Load the model in
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)  # Evaluate the model
print(str(accuracy * 100) + " Percent Accuracy")

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)
predictions = clf.predict(example_measures)
print(predictions)
