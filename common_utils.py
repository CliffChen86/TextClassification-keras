import numpy as np
import pandas as pd
import logging 
from keras.preprocessing.text import Tokenizer

def text_length_stat(texts=[],length_rate=0.98):
    length = []
    max_length = 0
    for text in texts:
        words = text.split(' ')
        length.append(len(words))
        if max_length < len(words):
            max_length = len(words)
            
    length = pd.DataFrame(length)
    
    res = int(length[0].quantile(length_rate))
    if res < 10:
        res = 10
    
    #printlog("real max sequence:" + str(max_length))
    #printlog("max_sequece_length:" + str(res))
    return res

def get_num_words(df_x ,min_frequency, is_word_freq = True):
    num_word = 0
    
    token = Tokenizer()
    token.fit_on_texts(df_x)
    
    word_counts = token.word_counts
    
    word_docs = token.word_docs
    
    if is_word_freq:
        for word in word_counts:
            if word_counts[word] >= min_frequency:
                num_words = num_word + 1
    else:
        for word in word_docs:
            if word_docs[word] >= min_frequency:
                num_words = num_words +1
    
    return num_words