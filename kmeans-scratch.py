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

# plt.scatter(X[:,0], X[:,1], s=150)
# plt.show()


# the colors used to label each different centroid class
colors = 10*["g", "r", "c", "b", "k"]


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
                self.classifications[i] = [] # kClass as key, feature sets as value

            for featureset in data:
                # calculate k number of average distances to centroids for one featureset
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                # Get index of which centroid is closer
                classification = distances.index(min(distances))
                # Append that point to the classification's index
                self.classifications[classification].append(featureset)
            # Getting ready to compare centroid change
            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                pass
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid)/original_centroid * 100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid)/original_centroid * 100.0))
                    optimized = False
            if optimized:
                break


    def predict(self, data):
        # calculate k number of average distances to centroids for one featureset
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        # Get index of which centroid is closer
        classification = distances.index(min(distances))
        return classification

clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

unknowns = np.array([[1, 2],
                    [12, 2],
                    [6, 1],
                    [3, 4],
                    [5, 6],
                    [8, 2]])
for unknown in unknowns:
    predictions = clf.predict(unknown)
    print(predictions)
    plt.scatter(unknown[0], unknown[1], marker="*", color=colors[predictions], linewidth=2)


plt.show()
