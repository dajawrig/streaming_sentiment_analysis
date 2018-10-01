# coding: utf-8

# In[1]:

import sys
reload(sys)
sys.setdefaultencoding('utf8')
from collections import Counter
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction import text as sk_fe_text
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn import grid_search
from sklearn.model_selection import *
from termcolor import *
import collections
import csv
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import pyprind
import re
import string
import seaborn as sns
import xgboost as xgb
from textblob import TextBlob

os.chdir("/home/dj/Desktop")

# In[6]:

infile = "tweet_sentiment.csv"
bloblist = list()

with open(infile, 'ab+') as csvfile:
    rows = csv.reader(csvfile)

    for row in rows:
        sentence = row[1]
        id = row[0]
        blob = TextBlob(sentence)
        bloblist.append((sentence, blob.sentiment.polarity, blob.sentiment.subjectivity))
        df_test = pd.DataFrame(bloblist, columns=['sentence', 'polarity', 'sentiment'])
        df_test.to_csv("df_test.csv")

print blob.sentiment.polarity, blob.sentiment.subjectivity
df_test.head()

# In[7]:

infile = "negative_tweets.csv"
bloblist = list()

with open(infile, 'ab+') as csvfile:
    rows = csv.reader(csvfile)

    for row in rows:
        sentence = row[1]
        id = row[0]
        blob = TextBlob(sentence)
        bloblist.append((sentence, blob.sentiment.polarity, blob.sentiment.subjectivity))
        train_neg = pd.DataFrame(bloblist, columns=['sentence', 'polarity', 'sentiment'])
        train_neg.to_csv("train_neg.csv")

print sentence
print blob.sentiment.polarity, blob.sentiment.subjectivity
train_neg.head()

# In[8]:

infile = "positive_tweets.csv"
bloblist = list()

with open(infile, 'ab+') as csvfile:
    rows = csv.reader(csvfile)

    for row in rows:
        sentence = row[1]
        id = row[0]
        blob = TextBlob(sentence)
        bloblist.append((sentence, blob.sentiment.polarity, blob.sentiment.subjectivity))
        train_pos = pd.DataFrame(bloblist, columns=['sentence', 'polarity', 'sentiment'])
        train_pos.to_csv("train_pos.csv")

print sentence
print blob.sentiment.polarity, blob.sentiment.subjectivity
train_pos.head()

# In[16]:

cols = ['id', 'sentence', 'sentiment', 'polarity']

positive_tweets = "positive_tweets.csv"
train_pos = pd.read_csv(positive_tweets, header=0, names=cols, encoding='utf-8', index_col=None)

negative_tweets = "negative_tweets.csv"
train_neg = pd.read_csv(negative_tweets, header=0, names=cols, encoding='utf-8', index_col=None)

df_train = pd.concat([train_neg, train_pos], axis=0, names=cols, join='outer')
df_train.to_csv("train_sentiment.csv")
df_train = pd.read_csv("train_sentiment.csv",
                       header=0, names=cols, error_bad_lines=False)

df_test = pd.read_csv('df_test.csv', names=cols, encoding='utf-8')

# In[12]:

df_train.head()

# In[18]:

print colored("Make things a little easier in case you don't have NLTK corpus for stopwords downloaded...", 'green')
print colored("Make sure to close NLTK corpus download popup box so code will continue running", 'red')
nltk.download()  # to download stopwords corpus; close popup box so code will continue running
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))
stops = (['''#',"u'",'@',"-","_","/","\","?","|"/"!",":",';','(','$',')', '[', ']','}'''])
tokenizer = RegexpTokenizer(r'\w+')
re.sub(r'[1,2,3,4,5,6,7,8,9,0]}\b', "", str(df_test[u'sentence']))
re.sub(r'\b\w{[https\?\:\/\/.*\/\*]}\b', "", str(df_test[u'sentence']))
re.sub(r'[' + string.punctuation + ']+', '', str(df_test['sentence']))
re.sub(r'\b\w{[@,-,_,/,?,|,!,:,;,\,(,$,),[,]+]}\b', "", str(df_test[u'sentence']))
text = tokenizer.tokenize(str(df_test[u'sentence']))

##### Section for linechart for most common words in tweets #####
print colored("Calculating term frequency...", 'green')

sns.set_style("whitegrid")
sns.set_palette("bright")

counter = Counter()
word = []
filtered = [word for word in text if (len(word) >= 5 and word not in stopWords and word not in stops and word != int)]
counts = Counter(filtered)

pbar = pyprind.ProgBar(len([filtered]),
                       title='Counting word occurrences...')
word_counts = sorted(counts, key=counts.get, reverse=True)
word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)}

counts.update(text)
print (counts.update(text))
pbar.update()

fdist = FreqDist(filtered)
vocab = fdist.keys()
print (vocab[:30])

mpl_fig = plt.figure(figsize=(12, 8))
ax = mpl_fig.add_subplot(111)

plt.title("Words with Highest Frequencies in Immunotherapy-related Tweets")
ax.set_xlabel('Term')
ax.set_ylabel('Frequency')
plt.xlabel("Term")
plt.ylabel("Frequency")

mpl_fig.tight_layout()
fdist.plot(40, cumulative=True)
mpl_fig.savefig("linechart.png")

######  Section for barchart for locations with most immunotherapy-related tweets ######
print colored("Visualizing locations with highest number of tweets...", 'blue')

mpl_fig = plt.figure(figsize=(12, 8))
ax = mpl_fig.add_subplot(111)

plt.title("Locations Posting Most Cancer Immunotherapy Tweets\n Based on 1000 Most Recent Tweets")
plt.xticks(rotation=90)

df = pd.read_csv("test_file.csv")
counts = df['Location'].value_counts()
x = counts.index
y = counts

sns.barplot(x, y)
plt.xlim([1, 20])
plt.ylim([1, 80])
ax.set_xlabel('Location')
ax.set_ylabel('Tweet Count')

plt.plot()
plt.show()
mpl_fig.savefig("barchart.png")

# In[131]:

###### Linechart showing Sentiment Scores and tweet frequency ######
import viscm

sns.set_style("whitegrid")
sns.set_palette("bright")

print colored("Visualizing sentiment results...", 'blue')

df = pd.read_csv("df_test.csv")
counts = df['polarity'].value_counts()
x = counts.index
y = counts

mpl_fig = plt.figure(figsize=(16, 8))
ax = mpl_fig.add_subplot(111)

plt.title("Sentiment Analysis Results:\n Negative = -1 to -0.1, Positive = 0.25 to 1, Neutral = -0.09 to 0.24")
plt.xticks(rotation=90)

sns.pointplot(x, y)

ax.set_xlabel('Sentiment Score')
ax.set_ylabel('Number of Tweets')

mpl_fig.tight_layout()
plt.plot()
plt.show()
mpl_fig.savefig("sentiment_barchart.png")

# In[123]:

print colored("Creating sentiment training model...", "blue")

def score (true, pred):
    return (precision_score(true, pred),
            recall_score(true, pred),
            f1_score(true, pred))

def print_score (s):
    print ("""
    Precision:  {:0.3}
    Recall:     {:0.3}
    F-Score:    {:0.3}
    """.format(*s))

neg = train_neg.sentence
pos = train_pos.sentence
# Create features and return sparse matricies
vectorizer = sk_fe_text.CountVectorizer(str(neg + pos))
vectorizer.fit(neg, pos)
neg_tdm = vectorizer.transform(neg).toarray()
pos_tdm = vectorizer.transform(pos).toarray()

# Create matricies and combine; transform negative tweets to 0s and positive tweets to 1
zeros = np.zeros((len(neg), 1))
ones = np.ones((len(pos), 1))
X = np.concatenate((neg_tdm, pos_tdm), axis=0)
y = np.ravel(np.concatenate((zeros, ones), axis=0))

# Create train/test split for modeling
trainX, testX, trainY, testY = train_test_split(X, y, train_size=.2)

# In[ ]:

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(trainX, trainY)

print("\n\nNaive Bayes Performance")
s = score(testY, nb.predict(testX))
print_score(s)

# In[ ]:

# SVM
from sklearn.svm import SVC

svm = SVC()
svm.fit(trainX, trainY)

print("\n\nSVM performance")
s = score(testY, svm.predict(testX))
print_score(s)

# In[ ]:

# Neural Network
from sklearn.neural_network import MLPClassifier

nn = MLPClassifier()
nn.fit(trainX, trainY)

print("\n\nNeural Network Performance")
s = score(testY, nn.predict(testX))
print_score(s)

# In[132]:

# Grid Search implemented to find best parameter combination
XGB_model = xgb.XGBClassifier(objective='binary:logitraw',
                              # type: XGBClassifier:gbtree, subsample=0.5,
                              n_jobs=3, nthread=4, colsample_bytree=0.5)
param_grid = {'max_depth': [4, 6], 'learning_rate': [0.1, 0.3]}
model = GridSearchCV(estimator=XGB_model,
                     param_grid=param_grid, scoring='accuracy', verbose=10,
                     iid=False, refit=True, cv=5)

print (model)
model.fit(X, y)
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.estimator.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Preparing test data for predictive analysis
df_test.set_index('id', inplace=True)
df_test = pd.merge(df_test.loc[:, ['id']], df_train, how='left',
                   left_index=True, right_index=True, sort=False)
X_test = test_X.fillna(-1)
id_test = df_test.index.values

y_pred = model.predict_proba(X_test)
df_train.set_index('id', inplace=True)
df_train = pd.concat([df_train['polarity'], df_test], axis=1, join='inner')

big_X = df_train[X].append(df_test[X])

big_X_imputed[X] = le.fit_transform(big_X_imputed[X])

for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

print ('Preparing inputs for the model')

train_X = big_X_imputed[0:df_train.shape[0]].as_matrix()
test_X = big_X_imputed[df_train.shape[0]::].as_matrix()
y_pred = model.predict_proba(y)

print ("Calculating most probable sentiment scores")
ids = []  # list of ids
cts = []  # list of sentiment_scores
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 1
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:4].tolist()

print "Printing results..."
df = pd.DataFrame(np.column_stack((ids, cts)), columns=['sentence', 'polarity'])
df.to_csv('results.csv', index=False)
print "Completed...files saved."
