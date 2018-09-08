import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans

style.use("ggplot")

# Input data for the model
X = np.array([[1, 2],
            [1.5, 1.8],
            [5, 8],
            [8, 8],
            [1, 0.6],
            [9, 11]])

# plt.scatter(X[:,0], X[:,1], s=150)
# plt.show()


# @param n_clusters for how many cluster groups to look for.
clf = KMeans(n_clusters=2)
# Model training on all data
clf.fit(X)

# The points in the X dimensions that denote where the centroids of the clusters are.
centroids = clf.cluster_centers_
# labels for each data point from X in order. Separated into 0 and 1.
labels = clf.labels_
# the colors used to label each different centroid class
colors = 10*["g.", "r.", "c.", "b.", "k."]
# for i in range(len(X)):
#     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
# plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5)
# plt.show()
