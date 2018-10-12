import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
import pandas as pd
from sklearn import preprocessing

# Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# survival Survival (0 = No; 1 = Yes)
# name Name
# sex Sex
# age Age
# sibsp Number of Siblings/Spouses Aboard
# parch Number of Parents/Children Aboard
# ticket Ticket Number
# fare Passenger Fare (British pound)
# cabin Cabin
# embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# boat Lifeboat
# body Body Identification Number
# home.dest Home/Destination

pd.set_option("display.max_columns", 1000)

df = pd.read_excel("titanic.xls")
original_df = pd.DataFrame.copy(df)

# Complete drop for these feature columns
df.drop(["body", "name", "boat"], 1, inplace=True)
df.convert_objects(convert_numeric=True) # Might be deprecated soon (@ convert_numeric)
df.fillna(0, inplace=True)


def handle_non_numerical_data(df):
    """
        Returns a dataframe with all data in a numerical format
        It is NOT one-hot-encoding. It just assigns 0,1,2,3... to each unique classe
    """
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {} # A dictionary { Class: to Number }
        def convert_to_int(val):
            return text_digit_vals[val]

        # Use on all non-numeric types of data
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column])) # Map class to number using original series column and mapping function convert_to_int
    return df

df = handle_non_numerical_data(df)
print(df.head())

X = np.array(df.drop(["survived"], 1).astype(float))
X = preprocessing.scale(X) # Boosts accuracy by a decent amount
Y = np.array(df["survived"])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df["cluster_group"] = np.nan

for i in range(len(X)):
    original_df["cluster_group"].iloc[i] = labels[i]
n_clusters_ = len(np.unique(labels))

survival_rates = {} # key is cluster group, value is survival rate
for i in range(n_clusters_):
    temp_df = original_df[ (original_df["cluster_group"] == float(i)) ] # a temporary df for one specific cluster
    survival_cluster = temp_df[ (temp_df["survived"] == 1) ]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)
