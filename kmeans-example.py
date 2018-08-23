import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans

style.use("ggplot")

X = np.array([[1, 2],
[1.5, 1.8],
[5, 8],
[8, 8],
[1, 0.6],
[9, 11]])
