import numpy as np
from sklearn.datasets.samples_generator import make_blobs,make_moons,make_circles,make_s_curve
from matplotlib import pyplot
from pandas import DataFrame
import matplotlib.pyplot as plt

#for reLU if the learning rate is very high, the parameters will not converge and jump to infinity

lam = 0.1 

np.random.seed(42)
#Defining model parameters and initializing them

def initialize_parameters_deep(layer_dims): #layer_dims is a list containing the number of hidden units in each layer.
	#np.random.seed(3)
	l = len(layer_dims)
	parameters = {} 
	for i in range(1,len(layer_dims)):
		parameters["W"+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])*np.sqrt(float(2)/layer_dims[i-1])		
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
	if (activation == "sigmoid"):
		A = 1/(1+np.exp(-Z))
		activation_cache = (A,Z)


	if (activation == "relu"):
		A = np.maximum(-0.0001,Z) #compares element-wise Z and broadcasted "0", and stores result in A
		activation_cache = (A,Z)
	
	cache = (linear_cache,activation_cache)
	return A,cache


def L_model_forward(X, parameters):
	#numbr of layers
	l = len(parameters)//2 #parameter is a dictionary with 2 parameters(W,b) for each layer, so this works	
	A_prev = X
	caches = []
	i=0
	#loop for l-1 layers using relu
	for i in range(1,l):
		A,cache = linear_activation_forward(A_prev,parameters["W"+str(i)],parameters["b"+str(i)],activation="relu")	
		A_prev = A
		caches.append(cache)

	#for lth layer using sigmoid
	AL,cache = linear_activation_forward(A_prev,parameters["W"+str(i+1)],parameters["b"+str(i+1)],activation="sigmoid")
	caches.append(cache)
	return AL,caches
	


#compute cost
def compute_cost(AL,Y,caches):
	#lam = 0.5 #hyper_param
	m = Y.shape[1]
	
	lg1 = np.log(AL)
	lg2 = np.log(1-AL)

	cost = (-float(1)/m)*np.sum((np.multiply(Y,lg1),np.multiply(1-Y,lg2)))  
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

def linear_activation_backward(dA,cache,activation):
	linear_cache,activation_cache = cache
	A,Z = activation_cache

	#compute dZ first
	if (activation == "relu"):
		dRebydZ = Z 
		dRebydZ[dRebydZ<0] = -0.0001 
		dRebydZ[dRebydZ>=0] = 0.9999
		dZ = np.multiply(dA,dRebydZ) 
	if (activation == "sigmoid"):
		dSigbydZ = np.multiply(A,1-A)
		dZ = np.multiply(dA,dSigbydZ)

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
	grads["dA" + str(l-1)], grads["dW" + str(l)], grads["db" + str(l)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
	
	
	 # Loop from l=L-2 to l=0
	for l in reversed(range(l-1)):

		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation = "relu")
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





def L_layer_model(X, Y, layers_dims, learning_rate = 0.2, num_iterations = 3000, print_cost=False):#lr was 0.2 for blobs and [2,2,1], lam =1 worked


	costs = []                         # keep track of cost


	parameters = initialize_parameters_deep(layers_dims)
    
    
	# Loop (gradient descent)
	for i in range(0, num_iterations):

		# Forward propagation: [LINEAR -> relu]*(L-1) -> LINEAR -> SIGMOID.

		AL, caches = L_model_forward(X, parameters)


		cost = compute_cost(AL, Y, caches)
         
    
		# Backward propagation.

		grads = L_model_backward(AL, Y, caches)
        
 
		# Update parameters.

		parameters = update_parameters(parameters, grads, learning_rate)
        
                
		# Print the cost every 100 training example
		if print_cost and i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))
		if print_cost and i % 100 == 0:
			costs.append(cost)
            
	# plot the cost
	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per tens)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()
    
	return parameters

def predict(X,parameters):
	AL,caches = L_model_forward(X, parameters)
	predictions = (AL>0.5)*1 #setting 0.5 as threshold and making predictions as 1 if greater than 0.5 and 0 otherwise.
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
	X = np.c_[xx.ravel(), yy.ravel()].T
	A = predict(X,parameters)
	Z = ((A)>0.5)*1
	fig = plt.figure()
	cols = color_set(Z[0,:])
	plt.scatter(xx, yy,c=cols,s=10, cmap=plt.cm.Spectral)
	plt.scatter(X_train[0,:], X_train[1,:],c=Y_train[0,:],s=10, cmap=plt.cm.Spectral)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	fig.savefig('blobs_relu_reg_2_10_10_5_1.png')



# generate 2d classification dataset
#X, y = make_s_curve(n_samples=100, noise=0.1) #not applicable
X, y = make_blobs(n_samples=5000, centers=2, n_features=2)
#X, y = make_moons(n_samples=5000, noise=0.2)
#X, y = make_circles(n_samples=5000, noise=0.05)
X_train = X.T
#print X_train.shape
Y_train = y.reshape(1,y.shape[0])
print X_train.shape,Y_train.shape


# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y',s=10, label=key, color=colors[key])
pyplot.show()


#final model

layers_dims = [2,10,10,5,1]#[2, 7,5,2, 1] 
parameters = L_layer_model(X_train, Y_train, layers_dims, num_iterations = 5000, print_cost = True)
#tomorrow
pred_train = predict(X_train, parameters)


# Print training accuracy
print ('Training Accuracy: %d' % float((np.dot(Y_train,pred_train.T) + np.dot(1-Y_train,1-pred_train.T))/float(Y_train.size)*100) + '%')


#pred_test = predict(test_x, test_y, parameters)
plot_decision_boundary(X_train,Y_train,parameters)

#take screen shot of the entire flowchart from coursera and apply financial aid





