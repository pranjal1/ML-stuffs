import tensorflow as tf
import numpy as np
from mnist import MNIST
from sklearn.preprocessing import LabelBinarizer
import math
from datetime import datetime


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs_added_fc2"
logdir = "{}/run-{}/".format(root_logdir, now)


#loading the training and testing data

mndata = MNIST('../Tensorflow/samples')

X_train, Y_train = mndata.load_training()
X_test, Y_test = mndata.load_testing()

X_train, Y_train  = np.array(X_train), np.array(Y_train).reshape(-1,1).T
X_test, Y_test  = np.array(X_test), np.array(Y_test).reshape(-1,1).T

#scaling
X_train = X_train/255.
X_test = X_test/255.

#one hot encoding of the outputs
onehot_encoder = LabelBinarizer()
onehot_encoded_train = onehot_encoder.fit_transform(Y_train.T)
onehot_encoded_test = onehot_encoder.fit_transform(Y_test.T)
Y_train = onehot_encoded_train
Y_test = onehot_encoded_test

print X_train.shape, X_test.shape
print Y_train.shape, Y_test.shape

n_epochs = 10

#mini-batch creation
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):


	#Creates a list of random minibatches from (X, Y)
    
	np.random.seed(seed)            # To make your "random" minibatches the same as ours
	m = X.shape[0]                  # number of training examples
 	mini_batches = []
        
	# Step 1: Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[permutation,:]
	shuffled_Y = Y[permutation,:]

	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		### START CODE HERE ### (approx. 2 lines)
		mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size,:]
		mini_batch_Y = shuffled_Y[k*mini_batch_size:(k+1)*mini_batch_size,:]
		### END CODE HERE ###
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
    
	return mini_batches


#initialize a placeholder for the minibatches

with tf.name_scope("inputs"):
	X = tf.placeholder(tf.float32,shape=(None,X_train.shape[1]),name = "X")
	X_reshaped = tf.reshape(X,shape=[-1,28,28,1])
	Y = tf.placeholder(tf.int32,shape=(None),name = "Y")
	training = tf.placeholder_with_default(False, shape=[], name='training') #dont know what for

conv1 = tf.layers.conv2d(X_reshaped,filters = 32,kernel_size = 3,strides = [1,1],padding = "SAME",activation = tf.nn.relu,name = "conv1")
conv2 = tf.layers.conv2d(conv1,filters = 64,kernel_size = 3,strides = [1,1],padding = "SAME",activation = tf.nn.relu,name = "conv2")

with tf.name_scope("pooling"):
	pool3 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1],padding = "VALID")
	pool3_flat = tf.reshape(pool3,shape = [-1,64*14*14]) #64 because, it is the number of channel (look number of filters in conv2), 7*7 will be size of individual feature matrix

with tf.name_scope("fc1"):
	fc1 = tf.layers.dense(pool3_flat, 64, activation=tf.nn.relu, name="fc1") #number of nodes in fc1

with tf.name_scope("fc2"):
	fc2 = tf.layers.dense(fc1, 20, activation=tf.nn.relu, name="fc2") #number of nodes in fc2

with tf.name_scope("output"):
	logits = tf.layers.dense(fc2, 10, name="output")#10 nodes in output layer, number of output classes
	Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
	xentropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y)#tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y)
	loss = tf.reduce_mean(xentropy)
	optimizer = tf.train.AdamOptimizer() 
	training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
	correct_prediction = tf.equal(tf.argmax(tf.transpose(logits)), tf.argmax(tf.transpose(Y)))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	#correct = tf.nn.in_top_k(logits, Y, 1)
	#accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

with tf.name_scope("for_tensor_board"):
	mse_summary = tf.summary.scalar('Xentropy', loss)
	file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())





with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		cntr = 0
		epoch_cost = 0
		minibatches = random_mini_batches(X_train, Y_train, mini_batch_size = 128, seed = epoch)
		for X_batch,Y_batch in minibatches:
			batch_loss,_ = sess.run([loss,training_op],feed_dict={X:X_batch,Y:Y_batch})
			epoch_cost += batch_loss / len(minibatches)
			if (cntr % 100 == 0):
				print(epoch, " epoch "," loop ",cntr,"batch loss ",batch_loss)
				sum_str = mse_summary.eval(feed_dict={X:X_batch,Y:Y_batch})
		   		file_writer.add_summary(sum_str)
			cntr += 1
		print "epoch_cost = "+str(epoch_cost)
		print "============================================================"
	file_writer.close()
        save_path = saver.save(sess, "./model_params/my_mnist_model")

#Note:tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels) expects logits of size (batch_size,features) and labels = (label,)


