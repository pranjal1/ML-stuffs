#in this task, I will create a data set (like a spiral dataset), and perform classification using 1-hidden layered neural network.
#if I can find a dataset in internet, I will use that as well.

#aim -> use scikit learn to load dataset.
#    -> perform classification using self built 1-layered NN.
#    -> perform visualization using matplotlib and scikit.



#Starting with the assumption that training and testing data are already provided in these numpy arrays.

#x_train.shape = (nx,m)
#y_train.shape = (1,m)

#x_test.shape = (p,nx)
#y_train.shape = (1,p)

#task tomorrow
#select a relevant dataset and perform the actual training and see the accuracies
#make a way to save the trained parameters and create an option to either train from scratch or make predictions using the saved parameters


#initializing the model layer parameters:

def layer_sizes(train_shape,output_shape): #(nx,m),(1,m) 
	n_x = train_shape[0]
	n_y = output_shape[0]
	return (n_x,n_y)


#initialize the weights and biases for all the layers

def initialize_parameters(n_x,n_h,n_y):
	W1 = np.random.randn(n_h,n_x) * 0.01
	b1 = np.zeros((n_h,1))
	W2 = np.random.randn(n_y,n_h) * 0.01
	b2 = np.zeros((n_y,1))
	
	parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}
	return parameters


#this function completes one complete forward propagation from input X to prediction Y
def forward_propagation(X,parameters): #this function will be called in a loop 
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	b1 = parameters["b1"]
	b2 = parameters["b2"]		
	
	Z1 = np.dot(W1,X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2,A1) + b2
	A2 = 1/(1+np.exp(-Z2))

	cache = {"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}
	return A2,cache
	

#this function will compute the cost
def compute_cost(A2,Y): #this function will be called once every 1000s like iteration to see how the cost is behaving
	m = Y.shape[1]
	
	log_part = -(np.multiply(Y,np.log(A2)) + np.multiply((1-Y),np.log(1-A2)))
	cost = (1/m) * np.sum(log_part)#this cost will be matrix form
	#to remove the matrix form
	cost = np.squeeze(cost)
	
	return cost



#perform the back propagation, to update the parameters
def back_propagation(parameters,cache,X,Y):
	m = Y.shape[1]
	W1 = parameters["W1"]
	W2 = parameters["W2"]

	A2 = cache["A2"]
	A1 = cache["A1"]

	dZ2 = A2-Y #since sigmoid activation in the last layer
	db2 = (1/m)*np.sum(dZ2)
	dW2 = (1/m) * np.dot(dZ2,A1.T)

	dZ1 = np.multiply(np.dot(W2.T,dZ2),(1-np.power(A1,2)))
	dW1 = (1/m)* np.dot(dZ1,X.T)
	db1 = (1/m)* np.sum(dZ1,axis=1,keepdims=True)

	grads = {"dW1":dW1,"dW2":dW2,"db1":db1,"db2":db2}
	return grads


#update parametes
def update_parameters(parameters,grads,learning_rate = 1.2): #learning_rate is tuned as a hyper parameter
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	b1 = parameters["b1"]
	b2 = parameters["b2"]

	dW1 = cache["dW1"]
	dW2 = cache["dW2"]
	db1 = cache["db1"]
	db2 = cache["db2"]

	W1 = W1 - learning_rate * dW1
	W2 = W1 - learning_rate * dW2
	b1 = W1 - learning_rate * db1
	b2 = W1 - learning_rate * db2
	
	parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}
	return parameters



def shallow_nn_model(X,Y,n_h,num_iterations = 10000,print_cost=False):
	np.random.seed(3) #for consistency
	#define layer parameters,n_h is already set by user
	n_x,n_y=layer_sizes(train_shape,output_shape)
	#initialize weights and biases for the first time
	parameters=initialize_parameters(n_x,n_h,n_y)
	
	#perform training	
	for i in range(num_iterations):
		#one forward pass
		A2,cache = forward_propagation(X,parameters)
		#compute cost
	 	cost = compute_cost(A2,Y)
		#compute update values for parameters
		grads = back_propagation(parameters,cache,X,Y)
		#perform update on parameters
		parameters = update_parameters(parameters,grads)
	
		if(i%1000 == 0 and print_cost):
			print "cost after "+str(i)+" iterations = "+str(cost)

	
	return parameters #to obtain the trained parameters for prediction 

	
def predict(X,parameters):
	A2,cache = forward_propagation(X,parameters)
	predictions = (A2>0.5)*1 #setting 0.5 as threshold and making predictions as 1 if greater than 0.5 and 0 otherwise.
	return predictions



#running the training 
parameters = shallow_nn_model(X_train,Y_train,n_h = 5,num_iterations = 10000,print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(5))


#make predictions
predictions = predict(X_train,parameters)
# Print training accuracy
print ('Training Accuracy: %d' % float((np.dot(Y_train,predictions.T) + np.dot(1-Y_train,1-predictions.T))/float(Y_train.size)*100) + '%')

#after this use a testing sample

#make predictions
predictions = predict(X_test,parameters)
# Print training accuracy
print ('Testing Accuracy: %d' % float((np.dot(Y_test,predictions.T) + np.dot(1-Y_test,1-predictions.T))/float(Y_test.size)*100) + '%')

