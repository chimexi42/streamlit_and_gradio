import os
import glob
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import  metrics
from sklearn.model_selection import train_test_split





print(os.listdir())

directory = r"C:\Users\Okoro Chima\Desktop\PYTHON PROJECTS"
directory2 = r"C:\Users\Okoro Chima\Desktop"
directory3 = r"C:\Users\Okoro Chima\Desktop\PYTHON PROJECTS\Titanic-App-master"
# PYTHON PROJECTS

for file in os.listdir():
    if 'csv' in file:
        print(file)


filenames = []
for root, dirs, files in os.walk(".", topdown=False):
    # print(root)
    # print(dirs)
    # print(files)

    for filename in files:
        if filename.endswith(".py"):
            filenames.append(filename)

print(filenames)

# print(os.listdir('/etc/'))

files = glob.glob('./*.csv')
# print(files)

# print([file.upper() for file in files])

# for file in files:
#     print(file)

# for file in files:
#     print("file name: ",file)
#     with open(file, "r") as f:
#         lines = f.readlines()
#         for line in lines:
#             print(line)

# BINARY SEARCH ALGORITHM

def binary_search(sequence, item):
    begin_index = 0
    end_index = len(sequence) -1
    while begin_index <= end_index:
        midpoint = begin_index + (end_index-begin_index) // 2
        midpoint_value = sequence[midpoint]


# Boosting ML
print()
dataset = pd.read_csv('mushrooms.csv')
print(dataset.head())
print(dataset.columns)

for label in dataset.columns:
    dataset[label] = LabelEncoder().fit_transform(dataset[label])
    print(dataset.head())

x = dataset.drop(columns = ['class'], axis = 1)
y = dataset['class']

print(dataset.info())
# print(x[:3].values)
# print(y[:3].values)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

model = DecisionTreeClassifier(criterion='entropy', max_depth=1)
adaboost = AdaBoostClassifier(base_estimator=model, n_estimators=100, learning_rate=1)

boostmodel = adaboost.fit(x_train, y_train)
y_pred = boostmodel.predict(x_test)

print(y_pred)
# print(model.score(x_test, y_test))

print("The accuracy score is ",metrics.accuracy_score(y_test, y_pred) * 100)

