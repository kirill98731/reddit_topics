"""Import required libraries"""
from gensim.models.fasttext import load_facebook_model
import compress_fasttext

"""Compression of a pre-trained model trained on Common Crawl and on Wikipedia using FastText"""
def model_compress():
    big_model = load_facebook_model('cc.en.300.bin').wv
    small_model = compress_fasttext.prune_ft_freq(big_model, pq=True)
    small_model.save('compressed.cc.en.300.bin')
