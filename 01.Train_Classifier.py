# -*- coding: utf-8 -*-
"""
@author: Samarth Bhasin
"""

# this script uses the function : review_to_words(raw_review)

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re

def review_to_words(raw_review):
    
    # Step 1: Clean HTML
    review_text = BeautifulSoup(raw_review).get_text() #single string (unicode)
    
    # Step 2: Clean ALL non-letters (including punctuations)
    letters_only = re.sub('[^a-zA-Z]',' ',review_text) #single string (unicode)
    
    # Step 3: Convert to lower case and split words
    lower_split = letters_only.lower().split() #list of strings (unicode)
    
    # Step 4: Convert stopwords list to set (searching in set is faster)
    stop = set(stopwords.words('english'))
    
    # Step 5: Remove stopwords
    meaningful_words = [r for r in lower_split if not r in stop] #list of strings (unicode)
    
    # Step 5: Construct sentence of single string
    return(" ".join(meaningful_words)) #single string (unicode)

###########################
# Reading Reviews Raw Data
###########################
reviews = pd.read_json('Tools_and_Home_Improvement_5.json',lines=True)
meta = pd.read_csv('THImetadata.csv')
fullDF = reviews.set_index('asin').join(meta.set_index('asin'),how='inner')
fullDF = fullDF.reset_index()
fullDF = fullDF[['asin','brand','reviewText','overall']]

train = fullDF.copy()

num_reviews = train['reviewText'].size


###################
# Cleaning Reviews
###################
clean_train_reviews = []
cou = 0
for i in train.index:
    cou += 1
    if cou%1000==0:
        print 'Review %d of %d ...' %(cou, num_reviews)
    clean_train_reviews.append(review_to_words(train['reviewText'][i]))

###############
# Bag of Words
###############
print('Creating the bag of words...\n')

from sklearn.feature_extraction.text import CountVectorizer
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag,count in zip(vocab,dist):
    print count,tag


print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["overall"] )