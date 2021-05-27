"""Import required libraries"""
import re
import string
from nltk.corpus import stopwords
stop_words = stopwords.words("english")

"""Text clearing function"""
def clear_text(x):
    x = x.lower()  # Lowercase the text
    x = ' '.join([word for word in x.split(' ') if word not in stop_words])  # Remove stop words
    x = x.encode('ascii', 'ignore').decode()  # Remove unicode characters
    x = re.sub(r'https*\S+', ' ', x)  # Remove URL
    x = re.sub(r'@\S+', ' ', x)  # Remove mentions
    x = re.sub(r'#\S+', ' ', x)  # Remove Hashtags
    x = re.sub(r'\'\w+', '', x)  # Remove ticks and the next character
    x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x)  # Remove punctuations
    x = re.sub(r'\w*\d+\w*', '', x)  # Remove numbers
    x = re.sub(r'\s{2,}', ' ', x)  # Replace the over spaces
    return x
