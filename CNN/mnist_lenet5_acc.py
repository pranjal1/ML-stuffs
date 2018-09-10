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

logits = tf.placeholder(tf.float32,shape = (None,10),name = "logits")
Y = tf.placeholder(tf.int32,shape = (None,10),name = "Y")
xentropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y)
 
with tf.Session() as sess:    
    saver = tf.train.import_meta_graph('./model_params/my_mnist_model.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model_params'))
    #coord = tf.train.Coordinator()
    #threads = []
    #for qr in sess.graph.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
    #    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
    #                                     start=True))
    #logits = sess.run('eval/Mean:0',feed_dict={"inputs/X:0":X_test[0:900,:],"inputs/Y:0":Y_test[0:900,]})
    logitx = sess.run('output/output/BiasAdd:0',feed_dict={"inputs/X:0":X_test[0:2,:]})
    print logitx
    print Y_test[0:2,:]
    print (sess.run(xentropy,feed_dict = {logits:logitx,Y:Y_test[0:2,:]})).shape
    print sess.run('train/softmax_cross_entropy_with_logits_sg/Reshape_2:0',feed_dict={"inputs/X:0":X_test[0:2,:],"inputs/Y:0":Y_test[0:2,:]})	
    print sess.run(argmax,feed_dict={X:logitx})	
    print sess.run(argmax,feed_dict={X:Y_test[0:2,:]})
