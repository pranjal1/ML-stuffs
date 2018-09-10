import tensorflow as tf
import numpy as np
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
from random import randint


import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning) #to avoid deprecation warnings

dataset = np.random.rand(64,28,28,1).astype(np.float32)
Y_tr = np.array([randint(0, 9) for p in range(0, 64)]).astype(np.int32).reshape(64,)
batch_size, height, width, channels = dataset.shape
print dataset.shape,Y_tr.shape



#X = tf.placeholder(tf.float32,shape=(None,X_batch.shape[1]),name = "X")
#X_reshaped = tf.reshape(X,shape=[batch_size,28,28,1])
X = tf.placeholder(tf.float32,shape=(None,height,width,channels))
Y = tf.placeholder(tf.int32,shape=(None))
conv1 = tf.layers.conv2d(X,filters = 32,kernel_size = 3,strides = [1,1],padding = "SAME",activation = tf.nn.relu,name = "conv1")
conv2 = tf.layers.conv2d(conv1,filters = 64,kernel_size = 3,strides = [1,1],padding = "SAME",activation = tf.nn.relu,name = "conv2")
pool3 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1],padding = "VALID")
pool3_flat = tf.reshape(pool3,shape = [-1,64*14*14])
fc1 = tf.layers.dense(pool3_flat, 64, activation=tf.nn.relu, name="fc1")
logits = tf.layers.dense(fc1, 10, name="output")#10 nodes in output layer, number of output classes
Y_proba = tf.nn.softmax(logits, name="Y_proba")
Y_proba_expanded = tf.expand_dims(Y_proba, 2)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y)
#loss = tf.reduce_mean(xentropy)
#optimizer = tf.train.AdamOptimizer() #takes default learning rate
#training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	output = sess.run(xentropy,feed_dict={X:dataset,Y:Y_tr})
	#output = sess.run(Y_proba_expanded,feed_dict={X:dataset})
	print dataset.shape
	print output.shape
