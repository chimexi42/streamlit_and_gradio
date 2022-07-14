import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
import random
from sklearn.tree import export_graphviz


x = [1,5,1.5, 8, 1, 9]
y = [2,8,1.8,8,0.6,11]

# plt.scatter(x,y)
# plt.show()

x = np.array([[1,2], [5,8], [1.5, 1.8], [8,8], [1,0.6], [9,11]])
# print(x)
plt.scatter(x[:, 0], x[:,1])
# plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(x)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

# colors = list("gcrcy")
colors = ['g', 'c', 'r', 'c', 'y']
print(colors)

for i in range(len(x)):
    print("coordinate: ", x[i], "label:",labels[i])
    plt.plot(x[i][0], x[i][1], colors[labels[i]], markersize = 10)

plt.scatter(centroids[:,0], centroids[:, 1], marker ="x", s=100, linewidths=5, zorder = 10)
# plt.show()
print()
# Autmation
for i in range(1,11):
    if i == 5:
        continue
    print(i)

print(string.ascii_letters)
print(string.digits)

# for _ in range(5):
#     password = random.choice(string.ascii_letters+string.digits)
#     print(password)
def func1():
    pass

print(func1())

def sum(*args):
    sum = 0
    for i in args:
        sum = sum +  i
    return sum

# print(sum(2,30,4,59))

# split and join
message = "hi how are you"
msg = message.split(" ")
print(msg)
msg_joined = "_".join(msg)
print(msg_joined)

# type conversion
value = 67
print(type(67))
value = str(value)
print(type(value))

# context manager

k = random.randint(1,5)
print(k)

list1 = [1,3,4,5,6,7,9]
print(list1[-1])

list2 = [2,3,4,5,0]
list2[0] = 0
print(list2)

print(False == False in[False])

number_grid = [
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [0]
]
for row in number_grid:
    for col in row:
        print(col)


# print(number_grid[0][0:2])
# print(number_grid[1][0:3])
# print(number_grid[3][0])
























