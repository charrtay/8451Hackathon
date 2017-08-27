import csv
import re
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn import svm

stop = set(stopwords.words('english'))
word_parser = RegexpTokenizer('[A-Za-z]+', flags=re.UNICODE)
digit_checker = re.compile("\d")

pd.options.mode.chained_assignment = None  # default='warn'
df = pd.read_csv('Reviews.csv')
sentiment = pd.read_csv('SentimentScore.csv')
df = df.merge(sentiment)

N = 100000
df_train = df.head(n=N)
df_test = df.tail(n=10000)

def clean(text):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.

    # 1. Remove HTML
    text = BeautifulSoup(text, "lxml").get_text()

    # 2. Convert words to lower case and remove non-letters
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'(\w+)s\b', r'\1', text)      # remove plurals
    text = re.sub(r'\b\w\w\b', '', text)         # remove two-letter "words"
    text = re.sub(r'from: .*@.*\b', '', text)    # remove email addresses
    text = re.sub(r'subject:', '', text)         # remove subject
    text = re.sub(r'\d+', '', text)              # remove digits
    text = re.sub(r'\d+\w+\d+', '', text)
    text = re.sub(r'\b\w+\d+', '', text)
    text = re.sub(r'\d+\w+\b', '', text)
    text = re.sub(r'\b_+([a-z]+)_+\b',r'\1', text)
    text = re.sub(r'\b([a-z]+)_+\b',r'\1', text)
    text = re.sub(r'\b_+([a-z]+)\b',r'\1', text)
    text = re.sub(r'\b_+\b','', text)

    # 3. Split string into words
    words = text.split()

    # 4. Remove stop words
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]

    # 5. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( words ))

# Clean
df['Cleaned'] = df['Text'].map(clean)

# Compute tf-idf on Reviews
tfidf_review = TfidfVectorizer(min_df=10, max_features=10000)
tfidf_review.fit(list(df['Cleaned']))

tfidf = dict(zip(tfidf_review.get_feature_names(), tfidf_review.idf_))
tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')
tfidf.columns = ['tfidf']

def has_non_ascii(review):
    try:
        review.decode('ascii')
        return 0
    except:
        return 1

def has_slash(review):
    if '\\' in review:
        return 1
    else:
        return 0

def has_duplicates(values):
    # For each element, check all following elements for a duplicate.
    for i in range(0, len(values)):
        for x in range(i + 1, len(values)):
            if values[i] == values[x]:
                return 1
    return 0

def calculate_tfidf_sum(tokens):
    score = 0
    for token in tokens:
        if tfidf.tfidf.get(token) is not None:
            score = score + tfidf.tfidf.get(token)
    return score

def calculate_tfidf_avg(tokens):
    score = 0
    for token in tokens:
        if tfidf.tfidf.get(token) is not None:
            score = score + tfidf.tfidf.get(token)
    if len(tokens)>1:
        mean = score/len(tokens)
    else:
        mean = 0
    return mean

def keyword_density(review):
    count = 0
    length = len(review)
    for word in review:
        if word in tfidf.index:
            count += 1
    density = count/length if count>1 else 0
    return density

def construct_features(data_frame):

    reviews = tfidf_review.transform(list(data_frame['Text']))
    data_frame['tokenized_review'] = data_frame['Text'].map(clean)

    # Compute title length
    if 'token_num' not in data_frame:
        data_frame['token_num'] = data_frame['tokenized_review'].map(len)

    # Compute new feature - 1 if title has duplicates, 0 if not.
    if 'has_duplicate' not in data_frame:
        data_frame['has_duplicate'] = data_frame['tokenized_review'].map(has_duplicates)

    # Compute tf-idf sum
    if 'tfidf_sum' not in data_frame:
        data_frame['tfidf_sum'] = data_frame['tokenized_review'].map(calculate_tfidf_sum)

    # Compute tf-idf average
    if 'tfidf_avg' not in data_frame:
        data_frame['tfidf_avg'] = data_frame['tokenized_review'].map(calculate_tfidf_avg)

    # Compute title length
    if 'review_length' not in data_frame:
        data_frame['review_length'] = data_frame['Text'].map(len)

    # Compute new feature - 1 if title has non-ascii character, 0 otherwise
    if 'has_non_ascii' not in data_frame:
        data_frame['has_non_ascii'] = data_frame['Text'].map(has_non_ascii)

    # Compute new feature - 1 if title has a backslash, 0 otherwise
    if 'has_backslash' not in data_frame:
        data_frame['has_backslash'] = data_frame['Text'].map(has_slash)

    X = data_frame[['token_num',
                  'has_duplicate',
                  'tfidf_sum',
                  'tfidf_avg',
                  'review_length',
                  'has_non_ascii',
                  'has_backslash']]

    X = np.concatenate([X.as_matrix(),
                        reviews.toarray(),
                        data_frame['Score'].as_matrix().reshape(-1,1),
                        data_frame['SentimentScore'].as_matrix().reshape(-1,1)
                       ], axis=1)

    return X

X = construct_features(df_train)
X.dump('features.pickle')

with open('vectors.pickle', 'rb') as file:
    v = pickle.load(file)

v = v[0:N]
X1 = np.concatenate((X, v), axis=1)

y = df_train['HelpfulnessNumerator']/df_train['HelpfulnessDenominator']
y = y.replace('NaN', 0)
y[y >= 0.25] = 1
y[y < 0.25] = 0
lab_enc = preprocessing.LabelEncoder()
y = lab_enc.fit_transform(y)

# SPLIT INTO TRAINING SET AND VALIDATION SET
X_train, X_validate, y_train, y_validate = train_test_split(X1, y, test_size=0.2)

# TRAIN AND EVALUATE THE MODEL
lgr = LogisticRegression()
lgr.fit(X_train, y_train)

#print("Model RMSE: %f" % mean_squared_error(lgr.predict_proba(X_validate)[:,1], y_validate)**0.5)

# Generate test features and evaluate model
d = {'0': {'Text': "I am very satisfied, product is as advertised, I use it on cereal, with raw vinegar, and as a general sweetner.", 'Score': 5, 'SentimentScore': 10}}
test_df = test_df.from_dict(d, orient='index', dtype=None)

model = Doc2Vec.load('reviews_vectors.mdl')
new_vector = model.infer_vector(test_df['Text'][0].split())
new_vector = new_vector.reshape(1,100)
test = construct_features(test_df)

test_features = np.concatenate((test, new_vector), axis=1)

print(lgr.predict_proba(test_features))
