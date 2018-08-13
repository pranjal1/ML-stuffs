#uses stochastic gradient descent to make predictions
#in batch gradient descent all the training tuples are used at once to update the model parameters (good convergence, high memory requirement)
#in stochastic gradient descent, one training tuple is used in one iteration to update the model parameters (high swings in model parameters, cannot gurantee convergence) (adv -> low memory requirement)
#middle ground -> mini batch gradient descent -> uses a subset of the training dataset to update the model parameters
from mnist import MNIST
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore",category=FutureWarning) 


#load the training and testing datasets

mndata = MNIST('samples')

X_train, Y_train = mndata.load_training()
X_test, Y_test = mndata.load_testing()


X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test = np.array(X_test)
Y_test = np.array(Y_test)


print X_train.shape,Y_train.shape
print X_test.shape,Y_test.shape

shuffle_index = np.random.permutation(60000)
X_train, Y_train = X_train[shuffle_index], Y_train[shuffle_index]

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, Y_train)

print("without scaling the cross-validation scores")
print cross_val_score(sgd_clf, X_train, Y_train, cv=3, scoring="accuracy")

#scaling will cause the mean to be 0 and standard deviation to be 1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

print("after scaling the cross-validation scores")
print cross_val_score(sgd_clf, X_train_scaled, Y_train, cv=3, scoring="accuracy")



#printing the confusion matrix for training dataset
#using the cross_val_predict, what this does is it determines the input tuple's output class from the model in which training is done using subsets of training set that does not contain input tuple.
Y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, Y_train, cv=3)
conf_mx = confusion_matrix(Y_train, Y_train_pred)
#print conf_mx

#plt.matshow(conf_mx, cmap=plt.cm.gray)
#plt.show()


#to highlight errors only
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx.astype(float)/row_sums
np.fill_diagonal(norm_conf_mx, 0)

#check the brightest spot, it represent classes with of maximum misclassification
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
