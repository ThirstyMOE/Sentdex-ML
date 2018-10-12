import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter
lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

'''
    Natural Language processing
'''
def create_lexicon(pog, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readLines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)  # Dictionary-like, key=word, value=count

    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:  # Filter ultra common and uncommon words
            l2.append(w)

    return l2

def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readLines()
        for l in content[:hm_lines]:
            current_words = word.tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features =
