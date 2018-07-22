###Requirements:
#Python -2.7.12
#Tensorflow -1.9.0
#Keras -2.2.0
#Numpy -1.14.5
#Scipy -1.1.0
#sklearn -0.19.2
#Pandas -0.23.3

# Main reference: https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/ 



#loading the libraries

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import pandas, numpy , string
from keras.preprocessing import text, sequence
from keras import layers,models, optimizers
from sklearn.metrics import f1_score,confusion_matrix
import matplotlib.pyplot as plt
from scipy import stats
from keras.models import load_model
import warnings


warnings.filterwarnings("ignore",category=DeprecationWarning) #to avoid deprecation warnings

#preparing the dataset

# loading the dataset
data = open('datasets/train_set.csv').read()
labels, text = [], []
for i, line in enumerate(data.split("\n")):
	content = line.split(",")
	labels.append(content[0])
	text.append(content[1])

#print(text.shape)

# using panda dataframe for easier visualization and manipulation
trainDF = pandas.DataFrame()
trainDF['text'] = text[1:] #removes the column header
trainDF['label'] = labels[1:] #removes the column header


# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'],test_size = 0.2,random_state=42, stratify=trainDF['label']) #20-80 split, stratified sampling



# to check if training and testing samples stratified
#print((train_y.value_counts()/train_y.size)*100)
#print((valid_y.value_counts()/valid_y.size)*100)



# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


#feature extraction

#using count vector as feature

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',encoding='latin-1') #default encoding of utf-8 is throwing error
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x) #scipy sparse csr_matrix
xvalid_count =  count_vect.transform(valid_x) #scipy sparse csr_matrix




#using tf-idf vector as feature

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',encoding='latin-1', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)


# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3),encoding='latin-1', max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)



# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3),encoding='latin-1', max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)




#function to build a given model, predict and compute accuracy

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
        
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    #print confusion_matrix(valid_y, predictions)
    print "f1 score = ", f1_score(valid_y, predictions, average='macro')*100		
    return metrics.accuracy_score(predictions, valid_y)*100,predictions


 

#function to print the accuracy using different features and ML models

def print_model_accuracies():
	print("Printing models F1 scores and accuracies...")
	print("================================================")
	pred_vector = numpy.zeros(valid_y.shape)

	# Naive Bayes on Count Vectors
	accuracy,predictions = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
	print "Naive Bayes, Count Vectors: ", accuracy
	print '\n'
	pred_vector=numpy.vstack([pred_vector,predictions])


	# Naive Bayes on tf-idf word-level Vectors
	accuracy,predictions = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
	print "Naive Bayes, tf-idf word-level Vectors: ", accuracy
	print '\n'
	pred_vector=numpy.vstack([pred_vector,predictions])

	# Naive Bayes on tf-idf ngram-level Vectors
	accuracy,predictions = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
	print "Naive Bayes, tf-idf ngram-level Vectors: ", accuracy
	print '\n'
	pred_vector=numpy.vstack([pred_vector,predictions])

	# Naive Bayes on tf-idf character-level Vectors
	accuracy,predictions = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
	print "Naive Bayes, tf-idf character-level Vectors: ", accuracy
	print '\n'
	pred_vector=numpy.vstack([pred_vector,predictions])



	# Linear Classifier on Count Vectors, saving this model due to good results
	accuracy,predictions = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
	print "Logistic Regression, Count Vectors: ", accuracy
	print '\n'
	pred_vector=numpy.vstack([pred_vector,predictions])

	# Linear Classifier on tf-idf word-level Vectors
	accuracy,predictions = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
	print "Logistic Regression, tf-idf word-level Vectors: ", accuracy
	print '\n'
	pred_vector=numpy.vstack([pred_vector,predictions])


	# Linear Classifier on tf-idf ngram-level Vectors
	accuracy,predictions = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
	print "Logistic Regression, tf-idf ngram-level Vectors: ", accuracy
	print '\n'
	pred_vector=numpy.vstack([pred_vector,predictions])

	# Linear Classifier on tf-idf character-level Vectors
	accuracy,predictions = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
	print "Logistic Regression, tf-idf character-level Vectors: ", accuracy
	print '\n'
	pred_vector=numpy.vstack([pred_vector,predictions])



	pred_vector = pred_vector[1:] 
	mode_prediction = stats.mode(pred_vector)[0]
	mode_prediction=mode_prediction.reshape(predictions.shape)
	print  "prediction accuracy based on mode from all other models: ", metrics.accuracy_score(valid_y, mode_prediction)*100
	print "f1 score for prediction based on mode = ", f1_score(valid_y, mode_prediction, average='macro')*100
	print "*************************************************************************"



#function to build a given model, predict and compute accuracy

def compute_labels_for_test(classifier, feature_vector_train, label):
	print "Predicting labels for test_set.csv ....."
	print "......"
	# fit the training dataset on the classifier
	classifier.fit(feature_vector_train, label)
	    
	#load the test data to predict the label
	data = open('datasets/test_set.csv').read()
	test_text = []
	for i,line in enumerate(data.split("\n")):
		test_text.append(line)   	
	test_text = test_text[1:] #removing column heading
	test_text_panda_series = pandas.Series(test_text)
	

	# transform the training and validation data using count vectorizer object
	xtest_count =  count_vect.transform(test_text_panda_series) #scipy sparse csr_matrix
		
	# predict the labels on validation dataset
	test_predictions = classifier.predict(xtest_count)
	if test_predictions.size>0:
		test_predictions_decoded = encoder.inverse_transform(test_predictions)
		prediction_to_file = pandas.DataFrame(test_predictions_decoded)
		prediction_to_file.columns = ['predicted labels']
		#print prediction_to_file.describe() #to see the predicted labels description
		prediction_to_file.to_csv("datasets/test_set_results.csv",index=False)
		print "Test results are saved as datasets/test_set_results.csv"
	else:
		print "No predictions!"


print_model_accuracies() #to compare the training accuracies and f1 score for the models
compute_labels_for_test(linear_model.LogisticRegression(), xtrain_count, train_y) #train and predict using the best model (currently Logistic Regression)


