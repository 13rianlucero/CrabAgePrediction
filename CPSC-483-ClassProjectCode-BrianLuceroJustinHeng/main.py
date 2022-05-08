import pandas
import numpy
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
X = data[["Length", "Diameter", "Height", "Weight", "Shucked Weight", "Viscera Weight", "Shell Weight"]]
y = data[["Age"]]

#Pearson correlation for every feature
col_cor = stats.pearsonr(data["SexValue"], y)
col1_cor = stats.pearsonr(data["Length"], y)
col2_cor = stats.pearsonr(data["Diameter"], y)
col3_cor = stats.pearsonr(data["Height"], y)
col4_cor = stats.pearsonr(data["Weight"], y)
col5_cor = stats.pearsonr(data["Shucked Weight"], y)
col6_cor = stats.pearsonr(data["Viscera Weight"], y)
col7_cor = stats.pearsonr(data["Shell Weight"], y)
print(col_cor)
print(col1_cor)
print(col2_cor)
print(col3_cor)
print(col4_cor)
print(col5_cor)
print(col6_cor)
print(col7_cor)

#split the data into test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=132)

#n_neighbors plot
error_rate = []
y_test2 = numpy.ravel(y_test)
for k in range(1, 31):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, numpy.ravel(y_train))
    knn_predict = neigh.predict(X_test)
    error_knn = 0
    for x in range(0, 1168):
        error_knn += abs(knn_predict[x] - y_test2[x])
    error_rate.append(error_knn/1169)

plt.plot(range(1, 31), error_rate)
plt.xlabel("n_neighbors")
plt.ylabel("error_rate")
plt.title("Average error vs n_neighbors")
plt.show()

#KNN
neigh = KNeighborsClassifier(n_neighbors=20)
neigh.fit(X_train, numpy.ravel(y_train))
knn_predict = neigh.predict(X_test)

#Multiple Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
score = r2_score(y_test,y_pred)


#SVR
regr = svm.SVR()
regr.fit(X_train, numpy.ravel(y_train))
regr_predict = regr.predict(X_test)

# #plot the predicted age against the actual age for the test set
plt.plot(range(1, 1169), knn_predict)
plt.plot(range(1, 1169), y_pred)
plt.plot(range(1, 1169), regr_predict)
plt.plot(range(1, 1169), numpy.ravel(y_test))
plt.xlim([0, 50])

#plt.xlim([60, 90])
plt.legend(["KNN Predicted Age", "LR Predicted Age", "SVR Predicted Age",  "Actual Age"])
plt.ylabel("Age in months")
plt.title("Predicted vs Actual Crab Age")
plt.show()

error_knn = 0
error_mlr = 0
error_svr = 0
y_test2 = numpy.ravel(y_test)
for x in range(0, 1168):
    error_knn += abs(knn_predict[x] - y_test2[x])
    error_mlr += abs(y_pred[x] - y_test2[x])
    error_svr += abs(regr_predict[x] - y_test2[x])

print (error_knn/1169)
print (error_mlr/1169)
print (error_svr/1169)