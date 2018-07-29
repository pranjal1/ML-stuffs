import numpy as np
from mnist import MNIST
from sklearn.preprocessing import LabelBinarizer
from sklearn import model_selection
import pandas


#load the training and testing datasets

mndata = MNIST('samples')

X, Y = mndata.load_training()
# or
#X_test, Y_test = mndata.load_testing()


# using panda dataframe for easier visualization and manipulation
trainDF = pandas.DataFrame()
trainDF['images'] = X
trainDF['label'] = Y

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['images'], trainDF['label'],test_size = 0.5,random_state=42, stratify=trainDF['label']) #20-80 split, stratified sampling




X_train = np.array([np.array(xi) for xi in train_x])/255.0
X_train = X_train.reshape(len(train_x[0]),train_x.shape[0])
Y_train = np.array(train_y)
Y_train = Y_train.reshape(1,Y_train.shape[0])
print X_train.shape,Y_train.shape



onehot_encoder = LabelBinarizer()
onehot_encoded = onehot_encoder.fit_transform(Y_train.T)
y_encoded = onehot_encoded.T

print y_encoded.shape



