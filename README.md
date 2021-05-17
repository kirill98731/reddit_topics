## Scraper Task 1
-------------------------------
- Scraper is based on [pushshift](https://github.com/pushshift/api) and [praw](https://praw.readthedocs.io/en/latest/code_overview/models/subreddit.html).
- Pandas, nltk were used to filter the data.
- As a result, 100174 documents were received.

__Date range__: 08-05-2019 - 20-04-2021

## Data preprocessing

Preprocessing steps:
- Lowercase the text
- Remove unicode characters
- Remove stop words
- Remove mentions
- Remove URL
- Remove Hashtags
- Remove ticks and the next character
- Remove punctuations
- Remove numbers
- Replace the over spaces

subreddits:
 - relationships
 - love
 - family
 - Marriage
 - Parenting
 - askwomenadvice
 - DecidingToBeBetter
 - depression
 - SuicideWatch
 - TwoXChromosomes

## EDA Task 2
Plots are in the `Task 2/figures` folder.  

## Features

I have chosen [fastText embeddings](https://fasttext.cc/docs/en/crawl-vectors.html) trained on Common Crawl and Wikipedia using fastText.

Original embeddings were pruned with this [lib](https://github.com/avidale/compress-fasttext).  
Pruned embeddings and all CSVs are available at my [Google Drive](https://drive.google.com/drive/folders/1fsIFOXNKdIvJV6pms2Vtr31b9q5U9xws?usp=sharing).  

## Tested Models Task 3

### Classification
 - SVC results without stemming/lemmatization ~ 0.72.  
 - LogisticRegression results without stemming/lemmatization ~ 0.73.  
 - LDA (discriminant analysis) results without stemming/lemmatization ~ 0.69.  

### Topic modeling
- LDA catched topics: Marriage, family, SuicideWatch, Parenting, relationship.

### Clustering
metric - [adjusted_rand_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
 - Kmeans using embeddings show bad results.
 - Kmeans using tfifd ~ 0.66 
 - MiniBatchKMeans using tfifd ~ 0.50
 
 ## Links to sourses
- [Scrap Reddit using pushshift](https://medium.com/@pasdan/how-to-scrap-reddit-using-pushshift-io-via-python-a3ebcc9b83f4)
- [Cleaning Text Data](https://towardsdatascience.com/cleaning-text-data-with-python-b69b47b97b76)
- [Feature Engineering - Handling Cyclical Features](http://blog.davidkaleko.com/feature-engineering-cyclical-features.html)
- [Topic modeling](https://webdevblog.ru/tematicheskoe-modelirovanie-s-pomoshhju-gensim-python/)