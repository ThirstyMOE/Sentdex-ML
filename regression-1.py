import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm # corss_validation is going to be deprecated soon
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

# Specify visual style of plot
style.use("ggplot")

data_frame = pd.read_csv("NASDAQOMX-XNDXT25NNR.csv")

data_frame = data_frame[["Index Value", "High", "Low", "Total Market Value", "Dividend Market Value"]]
data_frame["Index Over Total"] = data_frame["Index Value"] / data_frame["Total Market Value"]
data_frame = data_frame[["Total Market Value", "Index Value", "Low", "High"]]

forecast_col = "Total Market Value"
# Fill your NAN missing data with -99999
data_frame.fillna(-99999, inplace=True)

# Get the length of days out (0.01 for 10 days out in the future)
forecast_out = int(math.ceil(0.01*len(data_frame)))
print("We're looking out " + str(forecast_out) + " days")

# Shift the label data up the data_frame.
data_frame["label"] = data_frame[forecast_col].shift(-forecast_out)


# Capital X for features. This one drops the label column to get everything else
X = np.array(data_frame.drop(["label"], axis=1))
# scale down (normalize) the features. This takes big processing time
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

data_frame.dropna(inplace=True)
y = np.array(data_frame["label"])

# Split the data sets into train and test. Test set is 20% size
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Create sklearn LinearRegressor model or sklearn Support Vector Regression
clf = LinearRegression()
# clf = svm.SVR()

# Model training
clf.fit(X_train, y_train)

# open up a path for your pickle file (serialized model)
with open("linearregression.pickle", 'wb') as f:
    # Dump your model into the file
    pickle.dump(clf, f)
# Reopen your pickle model
pickle_in = open("linearregression.pickle", 'rb')
# Load the model in
clf = pickle.load(pickle_in)


# Model evaluation
accuracy = clf.score(X_test, y_test)
print(accuracy)
# Make a prediction with the future data
forecast_set = clf.predict(X_lately)
print(forecast_set)

#
# data_frame["Forecast"] = np.nan
#
# last_date = original_frame.iloc[-1].name
# print(last_date)
# last_unix = last_date.timestamp()
# one_day = 86400
# next_unix = last_unix + one_day
#
# for i in forecast_set:
#     next_date = datetime.datetime.fromtimestamp(next_unix)
#     next_unix += one_day
#     data_frame.loc[next_date] = [np.nan for _ in range(len(data_frame.columns) - 1)] + [i]
#
# data_frame["label"].plot()
# data_frame["Forecast"].plot()
# plt.legend(loc=4)
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.show()
