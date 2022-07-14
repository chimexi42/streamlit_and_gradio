import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn import svm
from sklearn.metrics import  accuracy_score
from sklearn.linear_model import  LinearRegression
import matplotlib.pyplot as plt

iris = datasets.load_iris()

x = iris.data
y = iris.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

classses = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']

model = svm.SVC()
model.fit(x_train, y_train)

print(model)
predictions = model.predict(x_test)

acc = accuracy_score(y_test, predictions)
print("Predictions",predictions)
print("actual",y_test)
print("Accuracy",acc)


for i in range(len(predictions)):
    print(classses[predictions[i]])

# Boston data
boston = datasets.load_boston()
x = boston.data
y = boston.target

print(x.shape)
print(y.shape)

print(x)
print(y)
model = LinearRegression()


plt.scatter(x.T[5], y)
# plt.show()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

model.fit(x_train, y_train)
predictions = model.predict(x_test)

print("predictions", predictions)
print("predictions", predictions)
print("score", model.score(x_test, y_test))
print("coefficient", model.coef_)
print("Intercept", model.intercept_)

# Kmeans

from sklearn.datasets import load_breast_cancer
from sklearn.cluster import  KMeans
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler

bc = load_breast_cancer()
print(bc)
x = scale(bc.data)
y = bc.target

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
model = KMeans(n_clusters=2, random_state=0)
model.fit(x_train)

predictions = model.predict(x_test)
labels = model.labels_


print("Labels", labels)
print("Actual", y_test)
print("Predictions",predictions)
print("Acc score", accuracy_score(y_test, predictions))

print(pd.crosstab(y_train, labels))
