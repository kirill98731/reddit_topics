"""Import required libraries"""
import pandas as pd
import numpy as np
import re

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
from sklearn import preprocessing

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from Task_3.utils.lemmatization import sent_to_words, lemmatization


"""Import data"""
df = pd.read_csv("final_embedded.csv", index_col=0)
df.head()

"""Division into training and test samples"""
X, y = df.drop(columns=['target']), df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

"""
Classification
SVC
"""
pipe = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('svc', SVC())
    ]
)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

"""LogisticRegression"""
pipe = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('svc', LogisticRegression())
    ]
)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

"""LDA"""
pipe = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('svc', LDA())
    ]
)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

"""
Topic modeling
LDA
Uploading wordnet
"""
nltk.download('wordnet')

"""Importing data"""
df = pd.read_csv("train.csv", index_col=0)
df_oh = df[['cleared_text', 'target']]

"""Lemmatization"""
lemmatizer = WordNetLemmatizer()
df_oh['cleared_text'] = df_oh['cleared_text'].map(lambda x: re.sub('\d+', '0', x))
df_oh['cleared_text'] = df_oh['cleared_text'].apply(lambda x: " ".join(lemmatizer.lemmatize(word) for word in x.split(" ")))

"""Word frequencies using TfidfVectorizer"""
vectorizer = TfidfVectorizer()
X_tf_idf = vectorizer.fit_transform(df_oh['cleared_text'].tolist())

"""LDA"""
lda = LatentDirichletAllocation(n_components=10, random_state=1)
lda.fit(X_tf_idf)

"""Top 10 words of each topic"""

vocab = vectorizer.get_feature_names()
n_top_words = 10
topic_words = {}
for topic, comp in enumerate(lda.components_):
    word_idx = np.argsort(comp)[::-1][:n_top_words]
    print([vocab[x] for x in word_idx],"\n")

"""
Gensim LDA
Clearing Text
"""

data_words = list(sent_to_words(df_oh['cleared_text']))

"""Building a bigram model"""
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


"""Lemmatization"""

data_words_bigrams = make_bigrams(data_words)
data_lemmatized = lemmatization(data_words_bigrams)

"""Creating a dictionary and corpus"""
id2word = corpora.Dictionary(data_lemmatized)
texts = data_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]

"""LDA"""
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=1000,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

"""Keywords for each topic and weight"""
lda_model.print_topics()


"""Clustering"""
le = preprocessing.LabelEncoder().fit(y)

"""Kmeans-fasttext"""
kmeans = KMeans(n_clusters=10).fit(X)
metrics.adjusted_rand_score(kmeans.predict(X), le.transform(y))

"""Kmeans-tfifd"""
km_tfidf=KMeans(n_clusters=10).fit(X_tf_idf)
metrics.adjusted_rand_score(km_tfidf.predict(X_tf_idf), le.transform(df_oh['target']))

"""MiniBatchKMeans"""
model = MiniBatchKMeans(n_clusters=10)
model.fit(X_tf_idf)
metrics.adjusted_rand_score(model.predict(X_tf_idf), le.transform(y))
