import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

housing = fetch_california_housing()
m, n = housing.data.shape

#construction of graph
#housing has features from x1 to xn, we add bias x0 (look linear regression tutorial)
housing_with_bias = np.c_[np.ones((m,1)),housing.data]

#input nodes
#apparently, we donot need to initialize the constants, initialization is only needed for variables
X = tf.constant(housing_with_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

#an important concept np.reshape(-1,1) means I want 1 columns, so figure out what should the number of rows be yourself
#np.reshape(2,-1) means I want 2 rows, so figure out what should the number of columns be yourself

#i will use transpose of X, so for this, I will create a node
XT = tf.transpose(X)

#the next node will compute the theta matrix,(this is needed to make predictions)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)

predictions = tf.matmul(X,theta)

#execution of graph
with tf.Session() as sess:
	theta_val = theta.eval()
	predictions = predictions.eval()



y_target = housing.target.reshape(-1, 1)
#print np.c_[predictions,y_target,predictions-y_target]
x_axis = np.arange(len(y_target))
plt.plot(x_axis, predictions-y_target, "r.")
plt.plot(x_axis, y_target, "b.")
plt.show()


