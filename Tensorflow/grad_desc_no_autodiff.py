import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
m, n = housing.data.shape

#construction of graph
#housing has features from x1 to xn, we add bias x0 (look linear regression tutorial)
housing_with_bias = np.c_[np.ones((m,1)),housing.data]

#scaling will cause the mean to be 0 and standard deviation to be 1.
scaler = StandardScaler()
housing_with_bias_scaled = scaler.fit_transform(housing_with_bias.astype(np.float64))

n_epochs = 1000
learning_rate = 0.01

#input nodes
#apparently, we donot need to initialize the constants, initialization is only needed for variables
X = tf.constant(housing_with_bias_scaled, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name="theta")

#i will use transpose of X, so for this, I will create a node
XT = tf.transpose(X)

y_pred = tf.matmul(X,theta,name="predictions")

error = y_pred - y

mse = tf.reduce_mean(tf.square(error),name="mse")

gradients = float(2)/m * tf.matmul(XT,error)

training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

#execution of graph
with tf.Session() as sess:
	sess.run(init)
	for x in range(0,n_epochs):
		if (x%100 == 0):
			print "MSE error = "+str(mse.eval())
		sess.run(training_op)
	best_theta = theta.eval()

print best_theta
