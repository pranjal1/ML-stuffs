import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from sklearn.preprocessing import LabelBinarizer
import math


#loading the training and testing data

mndata = MNIST('samples')

X_train, Y_train = mndata.load_training()
X_test, Y_test = mndata.load_testing()

X_train, Y_train  = np.array(X_train).T, np.array(Y_train).reshape(-1,1).T
X_test, Y_test  = np.array(X_test).T, np.array(Y_test).reshape(-1,1).T

#scaling
X_train = X_train/255.
X_test = X_test/255.

#one hot encoding of the outputs
onehot_encoder = LabelBinarizer()
onehot_encoded_train = onehot_encoder.fit_transform(Y_train.T)
onehot_encoded_test = onehot_encoder.fit_transform(Y_test.T)
Y_train = onehot_encoded_train.T
Y_test = onehot_encoded_test.T

print Y_train.shape, Y_test.shape



#mini-batch creation
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):


	#Creates a list of random minibatches from (X, Y)
    
	np.random.seed(seed)            # To make your "random" minibatches the same as ours
	m = X.shape[1]                  # number of training examples
 	mini_batches = []
        
	# Step 1: Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[:, permutation]
	shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		### START CODE HERE ### (approx. 2 lines)
		mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
		mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
		### END CODE HERE ###
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
    
	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		### START CODE HERE ### (approx. 2 lines)
		mini_batch_X = shuffled_X[:,num_complete_minibatches*mini_batch_size:]
		mini_batch_Y = shuffled_Y[:,num_complete_minibatches*mini_batch_size:]
		### END CODE HERE ###
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
    
	return mini_batches


#later see how to perform batch normalization in tensorflow

def one_hot_labels(labels,depth):
	one_hot_matrix = tf.one_hot(labels,depth,axis=0)
	return one_hot_matrix

def init_placeholders(nx,ny):
	X_batch = tf.placeholder(tf.float32,shape = [nx,None],name = "X_batch")
	Y_batch = tf.placeholder(tf.float32,shape = [ny,None],name = "X_batch")
	return (X_batch,Y_batch)

def initialize_parameters(shape_mtx):
	#shape_mtx = [X.shape[0],hl1,hl2,n_classes]
	parameters = {}
	W1 = tf.get_variable("W1",[shape_mtx[1],shape_mtx[0]],initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable("b1",[shape_mtx[1],1],initializer = tf.zeros_initializer())
	W2 = tf.get_variable("W2",[shape_mtx[2],shape_mtx[1]],initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable("b2",[shape_mtx[2],1],initializer = tf.zeros_initializer())
	W3 = tf.get_variable("W3",[shape_mtx[3],shape_mtx[2]],initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable("b3",[shape_mtx[3],1],initializer = tf.zeros_initializer())
	
	parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3}
	return parameters


def forward_propagation(X,parameters):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	W3 = parameters["W3"]
	b3 = parameters["b3"]
	
	#look to implement batch normalization
	Z1 = tf.add(tf.matmul(W1,X),b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2,A1),b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3,A2),b3)
	#remember we donot A3 as the cost function takes Z3 as parameter 
	
	return Z3

def compute_cost(Z3,Y):
	logits = tf.transpose(Z3)#this stupid transpose courtesy of following Andrew Ng :P
	labels = tf.transpose(Y)#this stupid transpose courtesy of following Andrew Ng :P
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
	return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,num_epochs = 100, minibatch_size = 64, print_cost = True):
	#load dimensions
	(n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
	n_y = Y_train.shape[0]                            # n_y : output size
	costs = []
	
	#initialize the placeholders
	X, Y = init_placeholders(n_x, n_y)
	
	#initialize parameters
	parameters = initialize_parameters([n_x,50,20,n_y]) #change the number of layers and number of nodes per layer

	#forward propagate
	Z3 = forward_propagation(X, parameters)	
	
	#compute cost
	cost = compute_cost(Z3,Y)
	
	#set optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	#initialize all the variables
	init = tf.global_variables_initializer()	 


	#random_mini_batches(X, Y, mini_batch_size = 64, seed = 0)
	
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			minibatches = random_mini_batches(X_train, Y_train, mini_batch_size = 64, seed = epoch)
			epoch_cost = 0
		
			for minibatch in minibatches:
				minibatch_X , minibatch_Y = minibatch

				_ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

				epoch_cost += minibatch_cost / len(minibatches)

			
			# Print the cost every epoch
			if print_cost == True and epoch % 1 == 0:
				print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
			if print_cost == True and epoch % 5 == 0:
				costs.append(epoch_cost)
			
		# plot the cost
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()

		# lets save the parameters in a variable
		parameters = sess.run(parameters)
		print ("Parameters have been trained!")

		# Calculate the correct predictions
		correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

		# Calculate accuracy on the test set
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
		print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
		return parameters


parameters = model(X_train, Y_train, X_test, Y_test)


