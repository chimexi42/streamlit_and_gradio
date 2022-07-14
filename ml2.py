from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans


iris = datasets.load_iris()
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
columns= iris['feature_names'] + ['species'])
# let's remove spaces from column name
iris.columns = iris.columns.str.replace(' ','')
print(iris.head())

x = iris.iloc[:, :3]
y = iris.species
sc = StandardScaler()
sc.fit(x)
x =sc.transform(x)

# print(x)

# K Means Cluster
model = KMeans(n_clusters=3, random_state=11)
model.fit(x)
print(model.labels_)

iris['pred_species'] = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)
print("Accuracy :", metrics.accuracy_score(iris.species, iris.pred_species))

# Set the size of the plot
plt.figure(figsize=(10,7))
# Create a colormap
colormap = np.array(['red', 'blue', 'green'])
# Plot Sepal
plt.subplot(2, 2, 1)
plt.scatter(iris['sepallength(cm)'], iris['sepalwidth(cm)'],c=iris.species, marker='o', s=50)
plt.xlabel('sepallength(cm)')
plt.ylabel('sepalwidth(cm)')
plt.title('Sepal (Actual)')
plt.subplot(2, 2, 2)
plt.scatter(iris['sepallength(cm)'], iris['sepalwidth(cm)'],c= iris.pred_species, marker='o', s=50)
plt.xlabel('sepallength(cm)')
plt.ylabel('sepalwidth(cm)')
plt.title('Sepal (Predicted)')
plt.subplot(2, 2, 3)
plt.scatter(iris['petallength(cm)'], iris['petalwidth(cm)'],c=iris.species,marker='o', s=50)
plt.xlabel('petallength(cm)')
plt.ylabel('petalwidth(cm)')
plt.title('Petal (Actual)')
plt.subplot(2, 2, 4)
plt.scatter(iris['petallength(cm)'], iris['petalwidth(cm)'],c= iris.pred_species,marker='o', s=50)
plt.xlabel('petallength(cm)')
plt.ylabel('petalwidth(cm)')
plt.title('Petal (Predicted)')
plt.tight_layout()
plt.show()


k = range(1,10)
km = [KMeans(n_clusters=k).fit(x) for k in K]
centroids = [k.cluster_centers_ for k in km]



