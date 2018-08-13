from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import numpy as np

def color_put(y):
	cols = []
	for x in y:
		if x==0:
			cols.append('Red')
		if x==1:
			cols.append('Green')
		if x==2:
			cols.append('Blue')
	return cols



iris = load_iris()
#X = iris.data #4 attributes
X = iris.data[:, 0:] # petal length and width
y = iris.target
print iris.feature_names[0:]


#best splitting factor in the first stage is petal length, check the plot of all features
f, ax = plt.subplots(2, 2, sharex='all', sharey='all')
cols = color_put(y)
ax[0,0].scatter(np.zeros(X[:,0].shape), X[:,0],c=cols,s=5, cmap=plt.cm.Spectral)
ax[0,1].scatter(np.zeros(X[:,1].shape), X[:,1],c=cols,s=5, cmap=plt.cm.Spectral)
ax[1,0].scatter(np.zeros(X[:,2].shape), X[:,2],c=cols,s=5, cmap=plt.cm.Spectral)
ax[1,1].scatter(np.zeros(X[:,3].shape), X[:,3],c=cols,s=5, cmap=plt.cm.Spectral)
plt.show()

'''
fig = plt.figure()
Axes3D = fig.add_subplot(111, projection='3d')
Axes3D.scatter(X[:,0],X[:,1],X[:,2],c=color_put(y),s=10)
plt.show()
'''

tree_clf = DecisionTreeClassifier(max_depth=4)#setting max_depth high, means overfitting 
tree_clf.fit(X, y)





export_graphviz(
        tree_clf,
        out_file="iris_tree1.dot",
        feature_names=iris.feature_names[0:],
        class_names=iris.target_names,
        rounded=True,
        filled=True)
