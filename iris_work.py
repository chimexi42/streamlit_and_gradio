from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz



iris = load_iris()

x = iris.data
y = iris.target
tree_clf = DecisionTreeClassifier()
model = tree_clf.fit(x,y)
dot_data = export_graphviz(tree_clf, out_file=None, feature_names=iris.feature_names, class_names= iris.target_names,
                           filled=True, rounded =True, special_characters= True)

graph = graphviz.Source(dot_data)
graph.render("iris")
# print(iris)
# print(x)
# print(y)






































