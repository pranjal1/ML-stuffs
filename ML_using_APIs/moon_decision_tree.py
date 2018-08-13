from sklearn.datasets.samples_generator import make_blobs,make_moons,make_circles,make_s_curve
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



X, y = make_moons(n_samples=5000, noise=0.1)


tree_clf = DecisionTreeClassifier(max_depth=8)#setting max_depth high, means overfitting 
tree_clf.fit(X, y)


export_graphviz(
        tree_clf,
        out_file="iris_moon.dot",
        feature_names=["X1","X2"],
        rounded=True,
        filled=True)

def color_set(Z):
	cols = []
	for x in Z:
		if x == 0:
			cols.append('Pink')
		if x == 1:
			cols.append('Black')
		if x == 2:
			cols.append('Yellow')
	return cols

def color_set1(Z):
	cols = []
	for x in Z:
		if x == 0:
			cols.append('Red')
		if x == 1:
			cols.append('Blue')
		if x == 2:
			cols.append('Green')
	return cols


def plot_decision_boundary(X,Y):
	# plot the resulting classifier


	h = 0.02
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		             np.arange(y_min, y_max, h))
	X_pred = np.c_[xx.ravel(), yy.ravel()]
	A = tree_clf.predict(X_pred)
	fig = plt.figure()
	cols = color_set(A)
	plt.scatter(xx, yy,c=cols,s=40, cmap=plt.cm.Spectral)
	cols = color_set1(Y)
	plt.scatter(X[:,0], X[:,1],c=cols,s=5, cmap=plt.cm.Spectral)
	plt.show()


plot_decision_boundary(X,y)
