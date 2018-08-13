from sklearn.datasets.samples_generator import make_blobs,make_moons,make_circles,make_s_curve
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


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
	A = polynomial_svm_clf.predict(X_pred)

	fig = plt.figure()
	cols = color_set(A)
	plt.scatter(xx, yy,c=cols,s=40, cmap=plt.cm.Spectral)
	cols = color_set1(Y)
	plt.scatter(X[:,0], X[:,1],c=cols,s=5, cmap=plt.cm.Spectral)
	#plt.xlim(xx.min(), xx.max())
	#plt.ylim(yy.min(), yy.max())
	plt.show()

def plot_features(X,Y):
	#x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	#y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	fig = plt.figure()
	#plt.scatter(X[:,0], X[:,1],c=color_set(Y),s=5, cmap=plt.cm.Spectral)
	X_new = X**3
	plt.scatter(X_new[:,0], X_new[:,1],c=color_set1(y),s=5, cmap=plt.cm.Spectral)
	#plt.xlim(x_min, x_max)
	#plt.ylim(y_min, y_max)
	plt.show()

'''
#for spiral_data
N = 500 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in xrange(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
'''

X, y = make_moons(n_samples=5000, noise=0.1)

polynomial_svm_clf = Pipeline((
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge"))
    ))

polynomial_svm_clf.fit(X, y)

plot_decision_boundary(X,y)

#plot_features(X,y)


