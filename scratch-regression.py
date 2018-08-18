from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# Stylin' your plot
style.use("fivethirtyeight")

# sample data
xs = [1, 2, 3, 4, 5, 6, 9, 12]
ys = [5, 4, 6, 5, 6, 7, 8, 5]

xs = np.array(xs, dtype=np.float64)
ys = np.array(ys, dtype=np.float64)

# plt.scatter(xs, ys)
# plt.show()
def best_fit_line(xs, ys):
    m = (mean(xs) * mean(ys)) - mean(xs * ys)
    m = m / ((mean(xs) ** 2) - (mean(xs ** 2)))
    b = mean(ys) - (m * mean(xs))
    return m, b

def predict(x, m, b):
    return ((m * x) + b)

m, b = best_fit_line(xs, ys)

regression_line = [((m * x) + b) for x in xs]

print(m)
print(b)
print(regression_line)

# Plot the regression_line
plt.plot(regression_line)
plt.scatter(13, predict(13, m, b), color="g")
# Plot the data
plt.scatter(xs, ys)
plt.show()
