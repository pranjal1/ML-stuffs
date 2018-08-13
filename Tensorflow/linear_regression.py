import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing


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


#execution of graph
with tf.Session() as sess:
	theta_val = theta.eval()

print theta_val
	
