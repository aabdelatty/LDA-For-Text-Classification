# LDA-For-Text-Classification
# Summary

In the last two decades, text classification got a lot of attention, and a lot of research have been investigating the development of better classifiers. Feature selection proved to be very critical for making the classification more efficient and accurate, which in turn led to intensively studies of text feature extraction. Latent Dirichlet allocation (LDA) is a generative statistical model, which is used as an unsupervised tool for topic modeling. LDA represents each document as a mixture of various topics, where there is a probabilistic distribution represents the probability of each word given each topic. LDA have been used as a future extraction tool for multiple NLP tasks like classification.  In this report, we are investigating the performance of LDA in text classification, and the factors upon which the performance depends. Moreover, the report is providing a comparison between LDA and the state of the art feature extraction techniques like tf-idf, and word to vector


# Datasets description

To compare the semantic extraction of each of our methods, the experiments have been carried out on two data sets R8 of Reuters 21578, and 20 newsgroupsges is 2,062 and labels are almost evenly distributed.

# Libraries used
1. sklearn: to use classifers like SVM, KNN, etc on both datasets.
2. nltk & gensim: for text preprocessing, feature extraction, train LDA model, etc..

# Runing Requirements
To run this classification tasks you need
1. Python version 3.52 or latter.
2. you need to have the following Python packages installed (sklearn, numpy, keras, and bs4)

# How to run
Both classification tasks have a main.py file that can be run directly. However, you may need to edit the list of tuning parameters.

In order to run the Word2Vec portion of the code you need to download Glove 100 dim (glove.6B.100d.txt) (can be downloaded from https://worksheets.codalab.org/bundles/0x15a09c8f74f94a20bec0b68a2e6703b3/) and save it in the same directory as the .py files
