"""Import required libraries"""
import math
import json
import requests
import time
from datetime import datetime, timedelta
import pandas as pd
from Task_1.data_preparation import clear_text

"""Data request function"""
def make_request(uri, max_retries=5):
    def fire_away(uri):
        response = requests.get(uri)
        assert response.status_code == 200
        return json.loads(response.content)

    current_tries = 1
    while current_tries < max_retries:
        try:
            time.sleep(1)
            response = fire_away(uri)
            return response
        except:
            time.sleep(1)
            current_tries += 1
    return fire_away(uri)


"""Collecting the required data"""
def map_posts(posts):
    return list(map(lambda post: {
        'title': post['title'],  # The title of the submission
        'score': post['score'],  # The number of upvotes for the submission
        'id': post['id'],  # ID of the submission
        'url': post['url'],  # The URL the submission links to
        'num_comments': post['num_comments'],  # The number of comments on the submission
        'created_utc': post['created_utc'],  # Time the submission was created
        'selftext': post.get('selftext', ''),  # The submissions’ selftext
        'author': post['author'],  # Provides an instance of Redditor
        'is_self': post['is_self'],  # Whether or not the submission is a selfpost (text-only)
        'subreddit': post['subreddit'],  # Provides an instance of Subreddit
        'cleared_text': '',  # Cleared selftext
        'link_flair_text': post.get('link_flair_text', '')  # The link flair’s text content
    }, posts))


"""Function for selecting documents with a number of characters greater than the threshold"""
def get_post(post_collections):
    post_collections_cleared = []
    for post in post_collections:
        if post['is_self'] and len(post['selftext'])>=2000:
            cleared_text = clear_text(post['selftext'])
            if len(cleared_text)>=2000:
                post['cleared_text'] = cleared_text
                post_collections_cleared.append(post)
    return post_collections_cleared


"""A function that returns all the documents of a certain community for a certain period"""
def pull_posts_for(subreddit, start_at, end_at):
    SIZE = 100
    URI_TEMPLATE = r'https://api.pushshift.io/reddit/search/submission?subreddit={}&after={}&before={}&size={}'
    it = 0
    post_collections = map_posts(make_request(URI_TEMPLATE.format(subreddit, start_at, end_at, SIZE))['data'])
    n_1 = len(post_collections)
    last = post_collections[-1]
    post_collections = get_post(post_collections)
    n_2 = len(post_collections)

    while n_1 == SIZE and n_2 <= 10000:
        new_start_at = last['created_utc'] + (5)
        more_posts = map_posts(make_request(URI_TEMPLATE.format(subreddit, new_start_at, end_at, SIZE))['data'])
        n_1 = len(more_posts)
        last = more_posts[-1]
        more_posts = get_post(more_posts)
        post_collections.extend(more_posts)
        n_2 = len(post_collections)
        it += 1
    print(it, n_1, n_2)
    return post_collections


"""Selected communities, because they are dominated by documents with a large number of characters"""
set_subreddit = [
    'relationships',
    'love',
    'family',
    'Marriage',
    'Parenting',
    'askwomenadvice',
    'DecidingToBeBetter',
    'depression',
    'SuicideWatch',
    'TwoXChromosomes'
]

"""The request returns no more than 100 documents, we make requests until we get the required number of documents"""
post_collections = []
end_at = math.ceil(datetime.utcnow().timestamp())
start_at = math.floor((datetime.utcnow() - timedelta(days=730)).timestamp())
URI_TEMPLATE = r'https://api.pushshift.io/reddit/search/submission?subreddit={}&after={}&before={}&size={}'
SIZE = 100
for subreddit in set_subreddit:
    it = 0
    n_3 = 0
    more_posts = map_posts(make_request(URI_TEMPLATE.format(subreddit, start_at, end_at, SIZE))['data'])
    n_1 = len(more_posts)
    last = more_posts[-1]
    n_temp = len(post_collections)
    post_collections.extend(get_post(more_posts))
    n_2 = len(post_collections)
    n_3+=(n_2-n_temp)
    while n_1 == SIZE and n_3 <= 10000:
        new_start_at = last['created_utc'] + 5
        more_posts = map_posts(make_request(URI_TEMPLATE.format(subreddit, new_start_at, end_at, SIZE))['data'])
        n_1 = len(more_posts)
        last = more_posts[-1]
        n_temp = len(post_collections)
        post_collections.extend(get_post(more_posts))
        n_2 = len(post_collections)
        n_3 += (n_2-n_temp)
        it += 1
    print(subreddit, it*SIZE+n_1, n_2, n_3)
posts_data = pd.DataFrame(post_collections)

"""Export data to csv"""
posts_data['title_cleared'] = posts_data['title'].apply(clear_text)
posts_data.to_csv("posts.csv")
