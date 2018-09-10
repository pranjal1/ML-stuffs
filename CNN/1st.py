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



#make filters, one horizontal feature extractor and another a vertical feature extractor

filters =  np.zeros((7,7,channels,2),dtype = np.float32)

#the shape may seem off.
#so the rule is for dataset the shape is (number of images,height,width,channels) -> (2,427,640,3)
#the shape of filter is (filter_height,filter_width,channels,number of filters) -> (7,7,3,2)
#the result is (number of images,new height,new width,number of filters) -> (2,214,320,2)

filters[3,:,:,0] = 1 #horizontal filter
filters[:,3,:,1] = 1 #vertical filter

#we have defined filters ourselves, which is not the practice in CNN

#print filters
X = tf.placeholder(tf.float32,shape=(None,height,width,channels))
convolution = tf.nn.conv2d(X,filters,strides=[1,3,3,1],padding = "SAME")

with tf.Session() as sess:
	output = sess.run(convolution,feed_dict={X:dataset})
	print dataset.shape
	print output.shape
