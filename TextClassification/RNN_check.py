#loading the libraries

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, numpy , string
# xgboost,textblob
from keras.preprocessing import text, sequence
from keras import layers,models, optimizers

import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.models import Model
from keras.callbacks import EarlyStopping

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
#print(((train_y.value_counts()/train_y.size)*100)
#print(((valid_y.value_counts()/valid_y.size)*100)



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
print((train_x[12481])
print((xtrain_count[1,:])
print( (count_vect.get_feature_names()[207],count_vect.get_feature_names()[2484],count_vect.get_feature_names()[4428],count_vect.get_feature_names()[6573],count_vect.get_feature_names()[7473],count_vect.get_feature_names()[7766],count_vect.get_feature_names()[8040],count_vect.get_feature_names()[8961],count_vect.get_feature_names()[10426],count_vect.get_feature_names()[11027],count_vect.get_feature_names()[13892])
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
lda_model = decomposition.LatentDirichletAllocation(n_components=70, learning_method='online', max_iter=20)
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


max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(train_x)
sequences = tok.texts_to_sequences(train_x)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)





#function to build a given model, predict and compute accuracy

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    print predictions		
    return metrics.accuracy_score(predictions, valid_y)


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



def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model



model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


train_y = train_y.reshape(-1,1)
valid_y = valid_y.reshape(-1,1) 

model.fit(sequences_matrix,train_y,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])



test_sequences = tok.texts_to_sequences(valid_x)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)


accr = model.evaluate(test_sequences_matrix,valid_y)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))



