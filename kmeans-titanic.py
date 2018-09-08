import matplotlib.pyplot as plt
import matplotlib import style
import numpy as np
import sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation

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
