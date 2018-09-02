# in this project, I will aim to perform MNIST handwriting classification using deep neural network from scratch step by step building helper functions.

#import necessary libraries

import numpy as np
from sklearn.datasets.samples_generator import make_blobs,make_moons,make_circles,make_s_curve
from matplotlib import pyplot
from pandas import DataFrame
import matplotlib.pyplot as plt
from mnist import MNIST
import pandas
from sklearn.preprocessing import LabelBinarizer
from sklearn import model_selection

lam = 0 

np.random.seed(42)
#Defining model parameters and initializing them

def initialize_parameters_deep(layer_dims): #layer_dims is a list containing the number of hidden units in each layer.
	#np.random.seed(3)
	l = len(layer_dims)
	parameters = {} 
	for i in range(1,len(layer_dims)):
		parameters["W"+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])*1		
		parameters["b"+str(i)] = np.zeros((layer_dims[i],1))
	return parameters	

#forward propagation helper functions

#computing the Z, caching the values A_prev,W,b as linear_cache for backward pass
#linear cache helps to compute the value of dW,db,dA given dZ
def linear_forward(A_prev,W,b):
	#print "A_prev shape = "+ str(A_prev.shape)
	#print "W shape = "+ str(W.shape)
	Z = np.dot(W,A_prev) + b
	cache = (A_prev,W,b)
	return Z,cache


#compute the value of a layer's activation A and cache Z,A as activation_cache for backward pass
#activation cache helps to compute dZ given dA
def linear_activation_forward(A_prev,W,b,activation):
	Z,linear_cache = linear_forward(A_prev,W,b)
	if (activation == "softmax"):
		A = np.apply_along_axis(lambda x: np.exp(x)/np.sum(np.exp(x)), 0, Z)
		activation_cache = (A,Z)


	if (activation == "tanh"):
		A = np.tanh(Z) #compares element-wise Z and broadcasted "0", and stores result in A
		activation_cache = (A,Z)
	
	cache = (linear_cache,activation_cache)
	return A,cache


def L_model_forward(X, parameters):
	#numbr of layers
	l = len(parameters)//2 #parameter is a dictionary with 2 parameters(W,b) for each layer, so this works	
	A_prev = X
	caches = []
	i=0
	#loop for l-1 layers using tanh
	for i in range(1,l):
		A,cache = linear_activation_forward(A_prev,parameters["W"+str(i)],parameters["b"+str(i)],activation="tanh")	
		A_prev = A
		caches.append(cache)

	#for lth layer using sigmoid
	AL,cache = linear_activation_forward(A_prev,parameters["W"+str(i+1)],parameters["b"+str(i+1)],activation="softmax")
	caches.append(cache)
	return AL,caches
	


#compute cost
def compute_cost(AL,Y,caches):
	#lam = 0.5 #hyper_param
	m = Y.shape[1]
	cost = (-float(1)/m)*np.sum((np.multiply(Y,np.log(AL)),np.multiply(1-Y,np.log(1-AL))))  
	cost = np.squeeze(cost)     #squeeze out the cost [[17]]--->17
	reg_loss = 0
	for i in caches:
		lin,act = i
		A,W,b = lin
		reg_loss = reg_loss + np.squeeze(np.sum(np.multiply(W,W)))
	#print "l2 = "+str(reg_loss)
	return cost+0.5*lam*reg_loss		




#at this point, we have completed the forward propagation for all the L-layers and also computed the cost for each iteration

#now we compute the gradients to update the model parameters W and b

#this function computes dW[l],db[l] and dA[l-1] given dZ[l] and (W,A[l-1]) from "linear cache"
def linear_backward(dZ,cache):#cache is linear_cache
	
	A_prev,W,b = cache
	m = A_prev.shape[1] #X.shape[1] = A_prev.shape[1] = m
	dW = (float(1)/m) * np.dot(dZ,A_prev.T)
	db = (float(1)/m) * np.sum(dZ,axis = 1, keepdims = True)
	dA_prev = np.dot(W.T,dZ)
		
	return dA_prev,dW,db

#to compute the above function, we need dZ, so we get it from this function	
#this function first computes dZ[l] given activation function and activation_cache to compute g'(Z[l]) and dA[l] from previous iteration of linear_backward and then linear_backward

def linear_activation_backward(dA,cache,activation):#cache is activation_cache
	linear_cache,activation_cache = cache
	A,Z = activation_cache

	#compute dZ first
	if (activation == "tanh"):
		#tanhval = np.tanh(Z)
		dZ = 1-np.power(A,2)
		dZ = np.multiply(dA,dZ) 
	if (activation == "softmax"):
		dSofbydZ = np.multiply(A,1-A)
		dZ = np.multiply(dA,dSofbydZ)

	#now compute linear_backward	
	dA_prev,dW,db = linear_backward(dZ,linear_cache)
	return dA_prev,dW,db



#backward propagation for all L layers
def L_model_backward(AL, Y, caches):
	
	grads = {} #main goal
	l = len(caches)
	m = AL.shape[1]	
	
	Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL, just checking
	
	#dAL = -Y/AL - (1-Y)/(1-AL)
	dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
	
	current_cache = caches[l-1]
	grads["dA" + str(l-1)], grads["dW" + str(l)], grads["db" + str(l)] = linear_activation_backward(dAL, current_cache, activation = "softmax")
	
	
	 # Loop from l=L-2 to l=0
	for l in reversed(range(l-1)):
		# lth layer: (tanh -> LINEAR) gradients.
		# Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 

		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation = "tanh")
		grads["dA" + str(l)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp


	return grads


#update parameters 
def update_parameters(parameters, grads, learning_rate):
	#lam = 0.75

	L = len(parameters) // 2 # number of layers in the neural network

	# Update rule for each parameter. Use a for loop.

	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)] - learning_rate * lam * parameters["W" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

	return parameters	





def L_layer_model(X_train, Y_train, layers_dims, learning_rate = 1.2, num_epochs = 100,batch_size =64, print_cost=False):
	costs = []
	parameters = initialize_parameters_deep(layers_dims)    	
    
	# Loop (gradient descent)
	for i in range(0, num_epochs):
		
		#for performing minibatch gradient descent
		shuffled_indices = np.random.permutation(Y_train.shape[1])

		X_train_shuffled =  X_train[:, shuffled_indices]
		Y_train_shuffled =  Y_train[:, shuffled_indices]
		y_encoded_shuffled =  y_encoded[:, shuffled_indices]


		for x in range(0,int(X_train.shape[1]/batch_size)):
			X = X_train_shuffled[:,x*batch_size:(x+1)*batch_size]	
			Y = Y_train_shuffled[:,x*batch_size:(x+1)*batch_size]
		

			# Forward propagation: [LINEAR -> tanh]*(L-1) -> LINEAR -> SIGMOID.

			AL, caches = L_model_forward(X, parameters)
		 
	    
			# Backward propagation.

			grads = L_model_backward(AL, Y, caches)
		
	 
			# Update parameters.

			parameters = update_parameters(parameters, grads, learning_rate)
		
		cost = compute_cost(AL, Y, caches)		
        	if print_cost:    
			print "cost after "+str(i+1)+" epoch = "+str(cost) 
		costs.append(cost)	
   	            
	# plot the cost
	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('epochs')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()
	return parameters

def predict(X,parameters):
	AL,caches = L_model_forward(X, parameters)
	predictions = onehot_encoder.inverse_transform(AL.T)
	return predictions




#load the training and testing datasets

mndata = MNIST('samples')

X, Y = mndata.load_training()
X_test, Y_test = mndata.load_testing()


X_train = (np.array(X).T)/255.0
Y_train = np.array(Y).reshape(1,len(Y))

X_test = (np.array(X_test).T)/255.0
Y_test = np.array(Y_test).reshape(1,len(Y_test))

#one hot encoding of the outputs
onehot_encoder = LabelBinarizer()
onehot_encoded = onehot_encoder.fit_transform(Y_train.T)
y_encoded = onehot_encoded.T




#final model

layers_dims = [X_train.shape[0],50,20,y_encoded.shape[0]] 
parameters = L_layer_model(X_train, y_encoded, layers_dims, num_epochs = 100,batch_size =64, print_cost = True)
#tomorrow
pred_train = predict(X_train, parameters)


# Print training accuracy
diff_mtx = pred_train - Y_train
diff_mtx = diff_mtx[np.where(diff_mtx == 0)]
print ('Training Accuracy: = '+ str(float(diff_mtx.size)/pred_train.size * 100)+'%')


pred_train = predict(X_test, parameters)
# Print testing accuracy
diff_mtx = pred_train - Y_test
diff_mtx = diff_mtx[np.where(diff_mtx == 0)]
print ('Testing Accuracy: = '+ str(float(diff_mtx.size)/pred_train.size * 100)+'%')

#save parameters
np.save('parameters_no_momentum.npy', parameters) 

# Load
#parameters = np.load('parameters_no_momentum.npy').item()



