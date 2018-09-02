import numpy as np
import tensorflow as tf

X = tf.constant(np.random.randn(3,1),name = "X")
W = tf.constant(np.random.randn(4,3),name = "W")
Y = tf.matmul(W,X)

with tf.Session() as sess:
	Y_eval = sess.run(Y)
	print Y_eval
	
