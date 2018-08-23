import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random
import pickle

# data: dicts with entry for each class
# predict for the prediction point
# k for the number of nearest neighbors to look for
def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("K is set to a value less than total voting groups!")

    distances = []
    for group in data:
        for features in data[group]:
            # Euclidean distance formula (by numpy)
            euclidean_distance = euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    # Get the classes for the top k (3) shortest distances)
    votes = [i[1] for i in sorted(distances)[:k]]
    # Return the top voted class
    vote_result = Counter(votes).most_common(1)[0][0]
    # Return the confidence of the top voted class
    confidence = (Counter(votes).most_common(1)[0][1] / k)
    return vote_result, confidence

accuracies = []
for i in range(25):
    data_frame = pd.read_csv("breast-cancer-wisconsin//breast-cancer-wisconsin.data.txt")

    data_frame.replace("?", -99999, inplace=True)  # Turn Missing Data into severe outliers
    data_frame.drop(["id"], axis=1, inplace=True)
    # Make sure the data is only ints or floats. (No quotes strings). Also change the data_frame to a list
    full_data = data_frame.astype(float).values.tolist()
    random.shuffle(full_data)

    test_size = 0.2

    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}

    train_data = full_data[:-int(test_size * len(full_data))]
    test_data = full_data[-int(test_size * len(full_data)):]

    for i in train_data:
        # i[-1] is the last column: the Class column (benign or malignant)
        # Append the rest of the list to the train_set's value list
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    # Iterate through each class
    for group in test_set:
        # Iterate through each data list in the test_set value list
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=15)
            if group == vote:
                correct += 1
            # else:
            #     print("Us", vote, "Confidence", confidence, "   Them", group)
            total += 1

    print("Accuracy:", correct/total)
    accuracies.append(correct/total)

print("Tested Accuracy:", sum(accuracies) / len(accuracies))

# dataset = {'k' : [[1, 2], [2, 3], [3, 1]],
#             'r' : [[6, 5], [7, 7], [8, 6]]}
# new_features = [5, 7]
# result = k_nearest_neighbors(dataset, new_features, k=3)
# print(result)

# style.use("fivethirtyeight")
# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color=i)
# plt.scatter(new_features[0], new_features[1], s=300, color=result)
# plt.show()
