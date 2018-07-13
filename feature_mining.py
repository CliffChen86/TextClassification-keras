import numpy as np
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd

class NNTokenPadding:
    #过滤掉符号，将每个词或者字映射成index
    def __init__(self, params, text_set):
        default_filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        self.tokenizer = Tokenizer(num_words = params.get('num_words'), filters = default_filter + params.get('word_filter',''))
        self.tokenizer.fit_on_texts(text_set)
        self.max_sequence_length = params.get('max_sequence_length',None)
    
    def extract(self, text_set):
        #将文本转化成index组成的序列
        sequence = self.tokenizer.texts_to_sequences(text_set)
        word_index = self.tokenizer.word_index
        data = pad_sequences(sequence, self.max_sequence_length)
        
        return data,word_index
    
class LabelCategorizer:
    def __init__(self, path = None):
        #create a empty dictionary
        self.label_map = dict() 
        self.label_re_map = dict()
        
    def to_category(self, label, num_class=None):
        #将数字label转换成one-hot形式
        return to_categorical(self.label_transform(label),num_class)
    
    def fit_on_labels(self, label):
        #value_counts()是一个计数方法，对出现的数字进行统计
        temp = pd.DataFrame(label)[0].value_counts().index  
        label_num = len(temp)
        
        for i in range(label_num):
            self.label_map[str(temp[i])] = str(i)
            self.label_re_map[str(i)] = str(temp[i])

    def label_transform(self, label):
        temp = [None] * len(label)
        for i in range(len(label)):
            temp[i] = self.label_map[str(label[i])]

        return temp

    def label_re_transform(self, label):
        temp = [None] * len(label)
        for i in range(len(label)):
            temp[i] = self.label_re_map[str(label[i])]

        return temp

    def save(self, path):
        with open(path, 'w') as f:
            f.write(str(len(self.label_map)) + ' ' + str(len(self.label_re_map)) + '\n')
            for key in self.label_map:
                f.write(key + ' ' + self.label_map[key] + '\n')
            for key in self.label_re_map:
                f.write(key + ' ' + self.label_re_map[key] + '\n')

    def load(self, path):
        lines = open(path).read().split('\n')
        length = lines[0].split(' ')
        length_label_map = int(length[0])
        length_label_re_map = int(length[1])

        self.label_map = dict()
        self.label_re_map = dict()
        for i in range(length_label_map):
            key_value = lines[i + 1].split(' ')
            self.label_map[key_value[0]] = key_value[1]
        for i in range(length_label_re_map):
            key_value = lines[i + 1 + length_label_map].split(' ')
            self.label_re_map[key_value[0]] = key_value[1]


class WordEmbedding:
    def __init__(self,params):
        self.tokenizer = Tokenizer()
        self.w2v_fp = params.get('w2v_fp', None)
    
    def extract(self, word_index):
        
        #printlog("loading word Embedding file:" + self.w2v_fp + "process started")
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(self.w2v_fp, binary = True)
        
        #embedding_dim = len(w2v_model["我"])
        #embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))
        embedding_matrix = np.zeros(w2v_model.syn0.shape[0] + 1,w2v_model.syn0.shape[1])
        
        not_in_model = 0
        in_model = 0
        for word, i in word_index.items():
            #对word_index中的每一条进行遍历,判断w2v里面的词是否在给的文本当中
            if word in w2v_model:
                in_model += 1
                embedding_matrix[i] = np.asarray(w2v_model[word], dtype = "float32")
                #词对应的序号的那一列更新词向量
            else:
                not_in_model += 1
        
        #printlog(str(in_model) + " words in " + self.w2v_fp)
        #printlog(str(not_in_model) + " words not in " + self.w2v_fp)
        
        #printlog("loading word embedding file: " + self.w2v_fp + " process finished")
        return embedding_matrix
