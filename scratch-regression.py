from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

# Stylin' your plot
style.use("fivethirtyeight")

# sample data
# xs = [1, 2, 3, 4, 5, 6, 7, 8]
# ys = [5, 4, 6, 5, 6, 7, 5, 9]
# xs = [1, 2, 3, 4, 5, 6]
# ys = [5, 4, 6, 5, 6, 7]

# xs = np.array(xs, dtype=np.float64)
# ys = np.array(ys, dtype=np.float64)

def create_dataset(number_of_points, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(number_of_points):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == "pos":
            val += step
        elif correlation and correlation == "neg":
            val -= step
    xs = [i for i in range(len(ys))]


    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

# plt.scatter(xs, ys)
# plt.show()
def best_fit_line(xs, ys):
    m = (mean(xs) * mean(ys)) - mean(xs * ys)
    m = m / ((mean(xs) ** 2) - (mean(xs ** 2)))
    b = mean(ys) - (m * mean(xs))
    return m, b

def predict(x, m, b):
    return ((m * x) + b)

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    se_reg = squared_error(ys_orig, ys_line)
    se_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (se_reg / se_mean)

xs, ys = create_dataset(40, 40, 2, correlation="neg")


m, b = best_fit_line(xs, ys)

regression_line = [((m * x) + b) for x in xs]

# print(m)
# print(b)
# print(regression_line)

x, y = 13, predict(13, m, b)

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

# Plot the regression_line
plt.plot(regression_line)
# plt.scatter(x, y, color="g")
# Plot the data
plt.scatter(xs, ys)
plt.show()
