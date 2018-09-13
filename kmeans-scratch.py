import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use("ggplot")

# Input data for the model
X = np.array([[1, 2],
            [1.5, 1.8],
            [5, 8],
            [8, 8],
            [1, 0.6],
            [9, 11]])

plt.scatter(X[:,0], X[:,1], s=150)
plt.show()


# the colors used to label each different centroid class
colors = 10*["g.", "r.", "c.", "b.", "k."]


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]
        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):

    def predict(self, data):
        pass
