import pandas
import numpy
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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
X = data[["SexValue", "Length", "Diameter", "Height", "Weight", "Shucked Weight", "Viscera Weight", "Shell Weight"]]
y = data[["Age"]]

#split the data into test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=132)

neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(X_train, numpy.ravel(y_train))

knn_predict = neigh.predict(X_test)

#plot the predicted age against the actual age for the test set
plt.plot(range(1, 1169), knn_predict)
plt.plot(range(1, 1169), numpy.ravel(y_test))
plt.xlim([0, 50])
plt.legend(["Predicted Age", "Actual Age"])
plt.ylabel("Age in months")
plt.title("Predicted vs Actual Crab Age")
plt.show()