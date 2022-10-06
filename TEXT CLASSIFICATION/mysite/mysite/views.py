#importing important libraries
import imp
from django.shortcuts import render
from django.http import HttpResponse

import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize

#tokenize the given data
data =" MY NAME IS JIGYASA"
print(word_tokenize(data))

from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
print(stopwords)

#delete punctuations

import string
data = "MY NAME IS JIGYASA"
data_1=[char for char in data if char not in string.punctuation]
print(data_1)

data_1=''.join(data_1)
print(data_1)

data_1 = data_1.split()
print(data_1)


#convert words into numbers i.e labelling

from sklearn.feature_extraction.text import CountVectorizer
data_1 = ["MY NAME IS JIGYASA"]

vectorizer = CountVectorizer()
vectorizer.fit(data_1)

print(vectorizer.vocabulary_)

vector = vectorizer.transform(data_1)
print(vector) 

 



