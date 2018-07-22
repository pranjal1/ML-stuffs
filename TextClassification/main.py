#loading the libraries

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, numpy , string
# xgboost,textblob
from keras.preprocessing import text, sequence
from keras import layers,models, optimizers
from sklearn.metrics import f1_score,confusion_matrix
import matplotlib.pyplot as plt
from scipy import stats


#preparing the dataset

# loading the dataset
data = open('datasets/train_set.csv').read()
labels, text = [], []
for i, line in enumerate(data.split("\n")):
	content = line.split(",")
	labels.append(content[0])
	text.append(content[1])


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

#to view the encoded y-labels
'''
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.h
ist(train_y,normed = True)
ax2 = fig.add_subplot(122)
ax2.hist(valid_y,normed = True)
plt.show()
'''


#at this point, we have prepared the training and testing labels.

#We cannot use whole of the training data to train the classifier, so we need to select most important feature from the training data.


#using count vector as feature

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',encoding='latin-1') #default encoding of utf-8 is throwing error
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x) #scipy sparse csr_matrix
xvalid_count =  count_vect.transform(valid_x) #scipy sparse csr_matrix


#to check if count vectorizer has correctly worked
'''
print(train_x[12481])
print(xtrain_count[1,:])
print (count_vect.get_feature_names()[207],count_vect.get_feature_names()[2484],count_vect.get_feature_names()[4428],count_vect.get_feature_names()[6573],count_vect.get_feature_names()[7473],count_vect.get_feature_names()[7766],count_vect.get_feature_names()[8040],count_vect.get_feature_names()[8961],count_vect.get_feature_names()[10426],count_vect.get_feature_names()[11027],count_vect.get_feature_names()[13892])
'''




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





'''
# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
X_train_topics = lda_model.fit_transform(xtrain_count) #(m,20)
X_valid_topics = lda_model.fit_transform(xvalid_count)


# view the topic models
topic_word = lda_model.components_ 
vocab = count_vect.get_feature_names()

n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
'''




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


def create_model_architecture(input_size):
    # create input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)
    
    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier 




pred_vector = numpy.zeros(valid_y.shape)

# Naive Bayes on Count Vectors
accuracy,predictions = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print "NB, Count Vectors: ", accuracy
print '\n'
pred_vector=numpy.vstack([pred_vector,predictions])


# Naive Bayes on tf-idf word-level Vectors
accuracy,predictions = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print "NB, tf-idf word-level Vectors: ", accuracy
print '\n'
pred_vector=numpy.vstack([pred_vector,predictions])

# Naive Bayes on tf-idf ngram-level Vectors
accuracy,predictions = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print "NB, tf-idf ngram-level Vectors: ", accuracy
print '\n'
pred_vector=numpy.vstack([pred_vector,predictions])

# Naive Bayes on tf-idf character-level Vectors
accuracy,predictions = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print "NB, tf-idf character-level Vectors: ", accuracy
print '\n'
pred_vector=numpy.vstack([pred_vector,predictions])

'''
# Naive Bayes on topic modeled Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), X_train_topics, train_y, X_valid_topics)
print "NB, topic modeled Vectors: ", accuracy
pred_vector=numpy.vstack([pred_vector,predictions])
'''
print"=========================================================="


# Linear Classifier on Count Vectors
accuracy,predictions = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print "LR, Count Vectors: ", accuracy
print '\n'
pred_vector=numpy.vstack([pred_vector,predictions])

# Linear Classifier on tf-idf word-level Vectors
accuracy,predictions = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print "LR, tf-idf word-level Vectors: ", accuracy
print '\n'
pred_vector=numpy.vstack([pred_vector,predictions])


# Linear Classifier on tf-idf ngram-level Vectors
accuracy,predictions = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print "LR, tf-idf ngram-level Vectors: ", accuracy
print '\n'
pred_vector=numpy.vstack([pred_vector,predictions])

# Linear Classifier on tf-idf character-level Vectors
accuracy,predictions = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print "LR, tf-idf character-level Vectors: ", accuracy
print '\n'
pred_vector=numpy.vstack([pred_vector,predictions])

'''
# Linear Classifier on topic modeled Vectors
accuracy,predictions = train_model(linear_model.LogisticRegression(), X_train_topics, train_y, X_valid_topics)
print "LR, topic modeled Vectors: ", accuracy
print '\n'
pred_vector=numpy.vstack([pred_vector,predictions])

print"=========================================================="

# SVM on Ngram Level TF IDF Vectors
accuracy,predictions = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print "SVM, N-Gram Vectors: ", accuracy
print '\n'
'''

print"=========================================================="
'''
classifier = create_model_architecture(X_train_topics.shape[1])
accuracy,predictions = train_model(classifier, X_train_topics, train_y, X_valid_topics, is_neural_net=True)
print "NN, count Vectors",  accuracy
print '\n'
print"=========================================================="
'''
pred_vector = pred_vector[1:] 
ans = stats.mode(pred_vector)[0]
print stats.mode(pred_vector)[1]
ans=ans.reshape(predictions.shape)
print metrics.accuracy_score(valid_y, ans)
print confusion_matrix(valid_y, ans)
print "f1 score = ", f1_score(valid_y, ans, average='macro')
