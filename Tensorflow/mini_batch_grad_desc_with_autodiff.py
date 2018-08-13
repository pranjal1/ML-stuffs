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
housing_with_bias_scaled = scaler.fit_transform(housing_with_bias.astype(np.float32))
y_target = housing.target.reshape(-1, 1).astype(np.float32)

n_epochs = 1
learning_rate = 0.01
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

#input nodes
#apparently, we donot need to initialize the constants, initialization is only needed for variables
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")




theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name="theta")

y_pred = tf.matmul(X,theta,name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name="mse")


#turns out there's an even effective way to reduce code
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)


init = tf.global_variables_initializer()


def fetch_batch(epoch, batch_index, batch_size):
	X_temp = housing_with_bias_scaled[batch_index*batch_size:(batch_index+1)*batch_size,:]
	y_temp = y_target[batch_index*batch_size:(batch_index+1)*batch_size,:]
	return X_temp,y_temp
	

#execution of graph
with tf.Session() as sess:
	sess.run(init)
	for epochs in range(n_epochs):	
		for batch_index in range(n_batches):
			#X_batch, y_batch = fetch_batch(epochs, batch_index, batch_size)	
			#sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
			if batch_index % 10 == 0:
				print "epoch "+str(epochs)+" error = "+str(mse.eval())
	
		best_theta = theta.eval()

print best_theta

