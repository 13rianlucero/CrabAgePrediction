import pandas
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

data = pandas.read_csv(r"CrabAgePrediction.csv").dropna(axis=0)
print(data.columns)
data["SexValue"] = 0 #create a new column

for index, row in data.iterrows():
    #convert male or female to a numerical value     Male=1, Female=2, Indeterminate=1.5
    if row["Sex"] == "M":
        data.iloc[index, 9] = 1
    elif row["Sex"] == "F":
        data.iloc[index, 9] = 2
    else:
        data.iloc[index, 9] = 1.5

#putting all our data together and dropping Sex for SexValue
data = data[["SexValue", "Length", "Diameter", "Height", "Weight", "Shucked Weight", "Viscera Weight", "Shell Weight", "Age"]]
print(data.head(10))