import tensorflow as tf
import numpy as np
from mnist import MNIST
from sklearn.preprocessing import LabelBinarizer
import math
from datetime import datetime




#loading the training and testing data

mndata = MNIST('../Tensorflow/samples')
X_test, Y_test = mndata.load_testing()
X_test, Y_test  = np.array(X_test).astype(np.float32), np.array(Y_test).astype(np.int32).reshape(-1,1).T

#scaling
X_test = X_test/255.

#one hot encoding of the outputs
onehot_encoder = LabelBinarizer()
onehot_encoded_test = onehot_encoder.fit_transform(Y_test.T)
Y_test = onehot_encoded_test.astype(np.int32)

print Y_test.shape, X_test.shape

X = tf.placeholder(tf.int32,shape = (None,10),name = "X")
argmax = tf.argmax(tf.transpose(X))

 
with tf.Session() as sess:    
    saver = tf.train.import_meta_graph('./model_params/my_mnist_model.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model_params'))
    logitx = sess.run('output/output/BiasAdd:0',feed_dict={"inputs/X:0":X_test[0:100,:]})	
    print sess.run(argmax,feed_dict={X:logitx})	
    print sess.run(argmax,feed_dict={X:Y_test[0:100,:]})

