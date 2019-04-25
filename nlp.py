# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 17:34:32 2018

@author: PAVEETHRAN
"""

#NLP

#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)#delimeter = tab..means..tsv file not csv..quoting here =3 means to ignore the double quotes

#cleaning the texts..
 import re
 import nltk
 from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
 nltk.download('stopwords')
 corpus=[]
 from nltk.corpus import stopwords
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])#removes all non alhapbetic characters and replaces them with spaces
    review=review.lower()
    review=review.split()#HERE THE DEFAULT DELIMITER IS SEPERATE BY SPACE ..NOT TAB..
    review=[word for word in review if not word in set(stopwords.words('english'))]#HERE IT MEANS REIVEIW =WORD..ONLY IF WORD NOT PRESENT N SPOTWORDS ..can also use "if any"..to take only those present/absent
    #HERE SET IS USED IN CASE WE USE TO COMPARE A ARTICLE OR ESSAY RATHER THAN A MERE LIST OF WORDS..ALSO FASTER IN PROCESSING
    review=[ps.stem(word) for word in review]
    #unicode is not wrong
    review=' '.join(review)#run this only once.
    corpus.append(review)
    
#CREATING BAG OF WORDS MODEL
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()#this is the independent variables(features) ,,,(x axis)
y=dataset.iloc[:,1].values

#WE USE NAIVE-BAYES FOR BETTER RESULTS

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


