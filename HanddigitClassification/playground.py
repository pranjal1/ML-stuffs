import numpy as np
from mnist import MNIST
from sklearn.preprocessing import LabelBinarizer
from sklearn import model_selection
import pandas


#load the training and testing datasets

mndata = MNIST('samples')

X, Y = mndata.load_training()
#X_test, Y_test = mndata.load_testing()


X_train = np.array(X).T
Y_train = np.array(Y).reshape(1,len(Y))



#one hot encoding of the outputs
onehot_encoder = LabelBinarizer()
onehot_encoded = onehot_encoder.fit_transform(Y_train.T)
y_encoded = onehot_encoded.T



#for performing minibatch gradient descent
shuffled_indices = np.random.permutation(Y_train.shape[1])

X_train_shuffled =  X_train[:, shuffled_indices]
Y_train_shuffled =  Y_train[:, shuffled_indices]
y_encoded_shuffled =  y_encoded[:, shuffled_indices]



batch_size = 64

for x in range(0,int(X_train.shape[1]/batch_size)):
	print X_train_shuffled[:,x*batch_size:(x+1)*batch_size].shape	
	print Y_train_shuffled[:,x*batch_size:(x+1)*batch_size].shape






