from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
import numpy as np
from nltk.corpus import reuters 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import timeit
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import Phrases
from gensim.corpora import Dictionary
stop_words = set(stopwords.words('english'))
p_stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

def word_transform(doc_set, ldamodel, dictionary,no_topics):
	data = []
	# loop through document list
	for i in doc_set:
		doc = ' '.join([word for word in i.split() if word not in stop_words])
		doc = doc.lower()
		doc = doc.split()
		#print(doc_lda)
		doc_prob = [0] * no_topics
		for j in doc:
			if j in stop_words:
				continue
			word_dist = ldamodel.get_document_topics(dictionary.doc2bow(list(j)))#ldamodel[dictionary.doc2bow(list(j))]
			for k in word_dist: 
				doc_prob[int(k[0])] = doc_prob[int(k[0])] + float(k[1])
		data.append(doc_prob )
		#print(doc_prob)
	return data

def topic_transform(doc_set, ldamodel, dictionary,no_topics):
	data = []
	# loop through document list
	for i in doc_set:
		doc = ' '.join([word for word in i.split() if word not in stop_words])
		doc = doc.lower()
		doc_lda = ldamodel[dictionary.doc2bow(tokenizer.tokenize(doc))]
		#print(doc_lda)
		doc_prob = [0] * no_topics
		for j in doc_lda:
			doc_prob[int(j[0])] = float(j[1])
		data.append(doc_prob)
		#print(doc_prob)
	return data
def tokenize(doc_set):
	# list for tokenized documents in loop
	texts = []
	tokenizer = RegexpTokenizer(r'\w+')
	# loop through document list
	for i in doc_set:
		# clean and tokenize document string
		raw = i.lower()
		tokens = tokenizer.tokenize(raw)

    	# remove stop words from tokens
		stopped_tokens = [i for i in tokens if not i in stop_words]
    
    	# stem tokens
		stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    	# add tokens to list
		texts.append(stemmed_tokens)
	return texts

def create_corpus(doc_set):
	texts = tokenize(doc_set)
	# turn our tokenized documents into a id <-> term dictionary
	dictionary = corpora.Dictionary(texts)
	dictionary.filter_extremes(no_below=10, no_above=0.5)
	# convert tokenized documents into a document-term matrix
	corpus = [dictionary.doc2bow(text) for text in texts]
	return dictionary, corpus

def load_glove_data():
	# read the word to vec file
	GLOVE_6B_100D_PATH = "glove.6B.100d.txt"
	dim = 100
	glove_small = {}
	with open(GLOVE_6B_100D_PATH, "rb") as infile:
		for line in infile:
			parts = line.split()
			try:
				word = parts[0].decode("utf-8")
				x = []
				for i in range(len(parts)-1):
					x.append(float(parts[i+1].decode("utf-8")))
				glove_small[word] = x
			except: 
				print('')
	return glove_small

def word2vec_transform(dataset, word2vec, dim):
	trans_data = []
	for doc in dataset:
		words = doc.lower().split()
		w_length = 1
		data = np.zeros(dim)
		for i in range(len(words)):
			if words[i] in word2vec and words[i] not in stop_words:
				data = data + word2vec[words[i]]
				w_length = w_length + 1
		data = data / float(w_length)
		trans_data.append(data)
	return trans_data

def load_data(path):
	X, y = [], []
	with open(path, "r") as infile:
		for line in infile:
			label, text = line.split("\t")
        	# texts are already tokenized, just split on space
        	# in a real case we would use e.g. spaCy for tokenization
        	# and maybe remove stopwords etc.
			X.append(text)
			y.append(label)
	return X, np.array(y)

def tf_classify(train_docs, train_labels, test_docs, test_labels):
	text_clf = Pipeline([('vect', CountVectorizer()), ('clf',KNeighborsClassifier(n_neighbors=1)) ])
	text_clf.fit(train_docs, train_labels) 
	predicted = text_clf.predict(test_docs)
	print("tf_KNN    " + str(100 * np.mean(predicted == test_labels)) )
	text_clf = Pipeline([('vect', CountVectorizer()), ('clf', SGDClassifier(random_state=42, tol=.001)), ])
	text_clf.fit(train_docs, train_labels) 
	predicted = text_clf.predict(test_docs)
	print("tf_SVM    " + str(100 * np.mean(predicted == test_labels))  )


def tf_idf_classify(train_docs, train_labels, test_docs, test_labels):
	text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',KNeighborsClassifier(n_neighbors=1)) ])
	text_clf.fit(train_docs, train_labels) 
	predicted = text_clf.predict(test_docs)
	print("tf_idf_KNN    " + str(100 * np.mean(predicted == test_labels)) )
	text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(random_state=42, tol=.001)), ])
	text_clf.fit(train_docs, train_labels) 
	predicted = text_clf.predict(test_docs)
	print("tf_idf_SVM    " + str(100 * np.mean(predicted == test_labels))  )

def word2vec_classify(train_docs, train_labels, test_docs, test_labels):
	word2vec = load_glove_data()
	train_docs = word2vec_transform(train_docs, word2vec, 100)
	test_docs = word2vec_transform(test_docs, word2vec, 100)
	text_clf = KNeighborsClassifier(n_neighbors=1)
	text_clf.fit(train_docs, train_labels) 
	predicted = text_clf.predict(test_docs)
	print("word2vec_KNN    " + str(100 * np.mean(predicted == test_labels)) )
	svm = SGDClassifier(random_state=420, tol=.001)#MultinomialNB()
	svm.fit(train_docs, train_labels)  
	predicted = svm.predict(test_docs)
	print("word2vec_SVM    " + str(100 * np.mean(predicted == test_labels))  )

def LDA_classify(train_docs, train_labels, test_docs, test_labels):
	text_clf = KNeighborsClassifier(n_neighbors=1)
	text_clf.fit(train_docs, train_labels) 
	predicted = text_clf.predict(test_docs)
	KNN_accu = 100 * np.mean(predicted == test_labels)
	svm = SGDClassifier(random_state=420, tol=.001)#MultinomialNB()
	svm.fit(train_docs, train_labels)  
	predicted = svm.predict(test_docs)
	SVM_accu = 100 * np.mean(predicted == test_labels)
	return KNN_accu,SVM_accu

def word_transform(doc_set, ldamodel, dictionary,no_topics):
	data = []
	# loop through document list
	for i in doc_set:
		doc = ' '.join([word for word in i.split() if word not in stop_words])
		doc = doc.lower()
		doc = doc.split()
		#print(doc_lda)
		doc_prob = [0] * no_topics
		for j in doc:
			if j in stop_words:
				continue
			word_dist = ldamodel.get_document_topics(dictionary.doc2bow(list(j)))#ldamodel[dictionary.doc2bow(list(j))]
			for k in word_dist: 
				doc_prob[int(k[0])] = doc_prob[int(k[0])] + float(k[1])
		data.append(doc_prob )
		#print(doc_prob)
	return data

train_data, train_labels = load_data("20ng-train-no-stop.txt")
test_data, test_labels = load_data("20ng-test-no-stop.txt")
tf_classify(train_data, train_labels, test_data, test_labels)
tf_idf_classify(train_data, train_labels, test_data, test_labels)
word2vec_classify(train_data, train_labels, test_data, test_labels)


no_topics = [5, 10, 15, 20, 25, 30]
dictionary, corpus = create_corpus(train_data + test_data)
time = np.zeros(len(no_topics))
topics_accu_KNN = np.zeros(len(no_topics))
topics_accu_SVM = np.zeros(len(no_topics))
words_accu_KNN = np.zeros(len(no_topics))
words_accu_SVM = np.zeros(len(no_topics))
for i in range(len(no_topics)):
	
	# train LDA model
	start = timeit.default_timer()
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=no_topics[i], id2word = dictionary, passes=30)
	end = timeit.default_timer()
	time[i] = end - start
	print(time[i])
	# get data represented as a distribution over the topics
	test_docs = topic_transform(test_data, ldamodel, dictionary,no_topics[i])
	train_docs = topic_transform(train_data, ldamodel, dictionary,no_topics[i])
	topics_accu_KNN[i], topics_accu_SVM[i] =  LDA_classify(train_docs, train_labels, test_docs, test_labels)
	test_docs = word_transform(test_data, ldamodel, dictionary,no_topics[i])
	train_docs = word_transform(train_data, ldamodel, dictionary,no_topics[i])
	words_accu_KNN[i], words_accu_SVM[i] =  LDA_classify(train_docs, train_labels, test_docs, test_labels)

plt.figure(1)
plt.plot(no_topics, time)
plt.xlabel('Number of Topics')
plt.ylabel('Time')

plt.figure(2)
plt.plot(no_topics, topics_accu_KNN)
plt.plot(no_topics, topics_accu_SVM)
plt.plot(no_topics, words_accu_KNN)
plt.plot(no_topics, words_accu_SVM)
plt.xlabel('Number of Topics')
plt.ylabel('Accuracy')
plt.legend(['KNN_topics', 'SVM_topics', 'KNN_words', 'SVM_words'], loc='upper left')
plt.show()
