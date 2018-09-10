import tensorflow as tf
import numpy as np
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt

import warnings


warnings.filterwarnings("ignore",category=DeprecationWarning) #to avoid deprecation warnings

china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
dataset = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = dataset.shape



#tf initialized the filters by itself
X = tf.placeholder(tf.float32,shape=(None,height,width,channels))
convolution = tf.layers.conv2d(X,filters=2,kernel_size=7,strides=[2,2],padding = "SAME") #tf.nn -> tf.layers
pooling = tf.nn.max_pool(convolution,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	output_conv,output_pool = sess.run([convolution,pooling],feed_dict={X:dataset})
	print output_conv.shape
	print output_pool.shape
	# plot 1st image's 1st feature map
	plt.imsave("conv_house.png", output_conv[0, :, :, 0])
	# plot 1st image's 1st feature map after pooling
	plt.imsave("conv_house_followed_by_pool.png", output_pool[0, :, :, 0])
	# plot 2st image's 1st feature map
	plt.imsave("conv_flower.png", output_conv[1, :, :, 0])
	# plot 2st image's 1st feature map after pooling
	plt.imsave("conv_flower_followed_by_pool.png", output_pool[1, :, :, 0])  


