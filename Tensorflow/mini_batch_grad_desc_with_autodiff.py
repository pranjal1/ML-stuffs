import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

housing = fetch_california_housing()
m, n = housing.data.shape

#construction of graph
#housing has features from x1 to xn, we add bias x0 (look linear regression tutorial)
housing_with_bias = np.c_[np.ones((m,1)),housing.data]

#scaling will cause the mean to be 0 and standard deviation to be 1.
scaler = StandardScaler()
housing_with_bias_scaled = scaler.fit_transform(housing_with_bias.astype(np.float32))
y_target = housing.target.reshape(-1, 1).astype(np.float32)

n_epochs = 10
learning_rate = 0.2
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

X = tf.placeholder(tf.float32, shape=(None, n+1),name = "X_pl")
y = tf.placeholder(tf.float32, shape=(None, 1),name = "y_pl")

theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name="theta")

y_pred = tf.matmul(X,theta,name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name="mse")


#turns out there's an even effective way to reduce code
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

y_training_acc = np.zeros(y_target.shape)

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

init = tf.global_variables_initializer()

def fetch_batch(epoch, batch_index, batch_size):
	shuff_indices = np.random.RandomState(epoch).permutation(m)
	housing_scaled_shuffled = housing_with_bias_scaled[shuff_indices]
	X_temp = housing_scaled_shuffled[batch_index*batch_size:(batch_index+1)*batch_size,:]
	y_temp = y_target[batch_index*batch_size:(batch_index+1)*batch_size,:]
	return X_temp,y_temp

#mini-batch is not a good choice here	


with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			#X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)		
			#sess.run(training_op, feed_dict={X: X_batch,y: y_batch})  # Will succeed.
			sess.run(training_op, feed_dict={X: housing_with_bias_scaled,y: y_target})  # Will succeed.
			if batch_index % 10 == 0:
					#summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
					summary_str = mse_summary.eval(feed_dict={X: housing_with_bias_scaled, y: y_target})
            				step = epoch * n_batches + batch_index
           				file_writer.add_summary(summary_str, step)
					print "epoch "+str(epoch)+" error = "+str(sess.run(mse, feed_dict={X: housing_with_bias_scaled,y: y_target}))
			
	y_training_acc = sess.run(y_pred,feed_dict={X: housing_with_bias_scaled})
	file_writer.close()


#print np.c_[y_training_acc,y_target,y_training_acc-y_target]
x_axis = np.arange(len(y_target))
plt.plot(x_axis, y_training_acc, "r.")
plt.plot(x_axis, y_target, "b.")
plt.show()
