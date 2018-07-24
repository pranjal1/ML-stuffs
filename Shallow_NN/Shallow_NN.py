#in this task, I will create a data set (like a spiral dataset), and perform classification using 1-hidden layered neural network.
#if I can find a dataset in internet, I will use that as well.

#aim -> use scikit learn to load dataset.
#    -> perform classification using self built 1-layered NN.
#    -> perform visualization using matplotlib and scikit.



import numpy as np
from sklearn.datasets.samples_generator import make_blobs,make_moons,make_circles,make_s_curve
from matplotlib import pyplot
from pandas import DataFrame
import matplotlib.pyplot as plt




# generate 2d classification dataset
#X, y = make_s_curve(n_samples=100, noise=0.1) #not applicable
X, y = make_blobs(n_samples=100, centers=2, n_features=2)
#X, y = make_moons(n_samples=5000, noise=0.1)
#X, y = make_circles(n_samples=10000, noise=0.05)
X_train = X.T
#print X_train.shape
Y_train = y.reshape(1,y.shape[0])
print X_train.shape,Y_train.shape
#print Y_train.shape


# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()







#initializing the model layer parameters:

def layer_sizes(train_shape,output_shape): #(nx,m),(1,m) 
	n_x = train_shape[0]
	n_y = output_shape[0]
	return (n_x,n_y)


#initialize the weights and biases for all the layers

def initialize_parameters(n_x,n_h,n_y):
	W1 = np.random.randn(n_h,n_x) * 1
	b1 = np.zeros((n_h,1))
	W2 = np.random.randn(n_y,n_h) * 1
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
	A2 = float(1)/(1+np.exp(-Z2))

	cache = {"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}
	return A2,cache
	

#this function will compute the cost
def compute_cost(A2,Y): #this function will be called once every 1000s like iteration to see how the cost is behaving
	m = Y.shape[1]
	
	log_part = -(np.multiply(Y,np.log(A2)) + np.multiply((1-Y),np.log(1-A2)))
	cost = (float(1)/m) * np.sum(log_part)#this cost will be matrix form
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
	db2 = (float(1)/m)*np.sum(dZ2)
	dW2 = (float(1)/m) * np.dot(dZ2,A1.T)

	dZ1 = np.multiply(np.dot(W2.T,dZ2),(1-np.power(A1,2)))
	dW1 = (float(1)/m)* np.dot(dZ1,X.T)
	db1 = (float(1)/m)* np.sum(dZ1,axis=1,keepdims=True)

	grads = {"dW1":dW1,"dW2":dW2,"db1":db1,"db2":db2}
	return grads


#update parametes
def update_parameters(parameters,grads,learning_rate = 1.2): #learning_rate is tuned as a hyper parameter
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	b1 = parameters["b1"]
	b2 = parameters["b2"]

	dW1 = grads["dW1"]
	dW2 = grads["dW2"]
	db1 = grads["db1"]
	db2 = grads["db2"]

	W1 = W1 - learning_rate * dW1
	W2 = W2 - learning_rate * dW2
	b1 = b1 - learning_rate * db1
	b2 = b2 - learning_rate * db2
	
	parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}
	return parameters



def shallow_nn_model(X,Y,n_h,num_iterations = 10000,print_cost=False):
	#print X.shape,Y.shape
	np.random.seed(3) #for consistency
	#define layer parameters,n_h is already set by user
	train_shape = X.shape
	output_shape = Y.shape
	n_x,n_y=layer_sizes(train_shape,output_shape)
	#initialize weights and biases for the first time
	parameters=initialize_parameters(n_x,n_h,n_y)
	#print parameters
	#perform training	
	for i in range(num_iterations+1):
		#one forward pass
		A2,cache = forward_propagation(X,parameters)
		#compute cost
	 	cost = compute_cost(A2,Y)
		#compute update values for parameters
		grads = back_propagation(parameters,cache,X,Y)
		#perform update on parameters
		parameters = update_parameters(parameters,grads)
	
		if(i%100 == 0 and print_cost):
			print "cost after "+str(i)+" iterations = "+str(cost)

	
	return parameters #to obtain the trained parameters for prediction 

	
def predict(X,parameters):
	A2,cache = forward_propagation(X,parameters)
	predictions = (A2>0.5)*1 #setting 0.5 as threshold and making predictions as 1 if greater than 0.5 and 0 otherwise.
	return predictions


def color_set(Z):
	cols = []
	for x in Z:
		if x == 0:
			cols.append('Pink')
		if x == 1:
			cols.append('Green')
	return cols

def plot_decision_boundary(X,Y,parameters):
	# plot the resulting classifier
	h = 0.02
	x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
	y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		             np.arange(y_min, y_max, h))
	Z = ((np.dot(parameters["W2"],np.dot(parameters["W1"],np.c_[xx.ravel(), yy.ravel()].T) + parameters["b1"]) + parameters["b2"])>0.5)*1
	fig = plt.figure()
	cols = color_set(Z[0,:])
	plt.scatter(xx, yy,c=cols,s=40, cmap=plt.cm.Spectral)
	plt.scatter(X_train[0,:], X_train[1,:],c=Y_train[0,:],s=40, cmap=plt.cm.Spectral)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	fig.savefig('boundary_4_neurons.png')



#running the training 
parameters = shallow_nn_model(X_train,Y_train,n_h = 4,num_iterations = 5000,print_cost=True)


#make predictions
predictions = predict(X_train,parameters)
# Print training accuracy
print ('Training Accuracy: %d' % float((np.dot(Y_train,predictions.T) + np.dot(1-Y_train,1-predictions.T))/float(Y_train.size)*100) + '%')


plot_decision_boundary(X_train,Y_train,parameters)




'''
#after this use a testing sample

#make predictions
predictions = predict(X_test,parameters)
# Print training accuracy
print ('Testing Accuracy: %d' % float((np.dot(Y_test,predictions.T) + np.dot(1-Y_test,1-predictions.T))/float(Y_test.size)*100) + '%')
'''
