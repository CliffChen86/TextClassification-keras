from keras.layers import Input,Embedding,Flatten,Concatenate,Dense,Lambda,Dropout
from keras.layers import Conv1D,MaxPooling1D
from keras.models import Model
from keras.models import load_model
from keras.callbacks import Callback, EarlyStopping
from keras import backend

from feature_mining import NNTokenPadding, LabelCategorizer, WordEmbedding
from common_utils import text_length_stat

import numpy as np
import time
import pickle

class TextCNN:
    def __init__(self, params):
        #printlogInit()
        self.model_name = 'CNN'
        self.framework_name = 'keras'

        #w2v parameters
        self.default_w2v_fp = params.get('default_w2v_fp', None)
        self.w2v_fp = params.get('w2v_fp','default')

        if self.w2v_fp != None:                    # if there exist a w2v
            if self.w2v_fp == 'default':
                self.w2v_fp = self.default_w2v_fp
            self.use_external_embedding = True
        else:
            self.use_external_embedding = False

        #layer parameters
        self.embedding_dim = int(params.get('embedding_dim', 300))
        self.use_external_embedding = bool(params.get('use_external_embedding', False))
        self.embedding_trainable = bool(params.get('embedding_trainable', True))
        self.dropout_rate = float(params.get('dropout_rate', 0.5))
        self.filter_num = int(params.get('filter_num', 128))

        #self.filter_sizes = map(int, params.get('filter_size','3,4,5').split(','))
        filter_size = params.get('filter_size', '3,4,5').split(',')
        temp = []
        for i in filter_size:
            temp.append(int(i))
        self.filter_sizes = temp
        self.conv_activation = params.get('conv_activation', 'tanh')
        self.conv_strides = int(params.get('conv_strides', 1))
        self.conv_padding = params.get('conv_padding', 'valid')

        pooling_sizes = params.get('pooling_size', '3,4,5').split(',')
        temp = []
        for i in pooling_sizes:
            temp.append(int(i))
        self.pooing_sizes = temp

        pooling_strides = params.get('pooing_strides', '3,4,5').split(',')
        temp = []
        for i in pooling_strides:
            temp.append(int(i))
        self.pooling_strides = temp

        self.pooing_padding = params.get('pooing_padding','valid')
        self.dense_size = int(params.get('dense_size',256))
        self.dense_activation = params.get('dense_activation','tanh')

        self.model_loss = params.get('model_loss', 'categorical_crossentropy')
        self.model_optimizer = params.get('model_optimizer', 'Adadelta')

        self.model_metrics = params.get('model_metrics', ['accuracy'])
        self.model_epoch = params.get('model_epoch', 5)
        self.model_train_batchsize = params.get('model_train_batchsize', 128)
        self.model_test_batchsize = params.get('model_test_batchsize', 1024)

        #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "initialization finished")
    
    def construct_graph(self, embedding_matrix = None):
        input_layer = Input(shape=(self.max_sequence_length, ))
        
        if self.use_external_embedding :
            assert embedding_matrix is not None
            embedding_layer = Embedding(self.vocab_size,
                                        self.embedding_dim,
                                        weights = [embedding_matrix],
                                        input_length =self.max_sequence_length,
                                        trainable = self.embedding_trainable)(input_layer)
        else:
            embedding_layer = Embedding(self.vocab_size,
                                        self.embedding_dim,
                                        input_length = self.max_sequence_length,
                                        trainable=self.embedding_trainable)(input_layer)
        
        # get conv-pool block for each filter size
        conv_blocks = []
        for i in range(len(self.filter_sizes)):
            conv = Conv1D(filters=self.filter_num,
                          kernel_size = self.filter_sizes[i],
                          strides = self.conv_strides,
                          padding = self.conv_padding,
                          activation = self.conv_activation)(embedding_layer)
            
            maxpool = MaxPooling1D(pool_size = self.pooing_sizes[i],
                                 strides = self.pooling_strides[i],
                                 padding = self.pooing_padding)(conv)
            
            flatten = Flatten()(maxpool)
            
            conv_blocks.append(flatten)
        
        #4 combine all flatten conv_pool feature
        concatente_layer = Concatenate()(conv_blocks)
        
        #5 add dorpout
        dropout_layer = Dropout(self.dropout_rate)(concatente_layer)
        
        #6 add dense layer and output layer
        dense_layer = Dense(units = self.dense_size,
                           activation = self.dense_activation)(dropout_layer)
        
        output_layer = Dense(units = self.label_num,activation = 'softmax')(dense_layer)
        
        self.model = Model(inputs = input_layer,outputs = output_layer)
        
        #printlog(self.model.summary())
        
    def train(self, x, y, *, val_x=None, val_y=None): 
        #printlog(self.get_framework_name() + " " + self.get_model_name())
        train_start_time = time.time()
        
        df_x = x
        df_y = y
        if val_x is not None and val_y is not None: 
            df_x = x + val_x
            df_y = y + val_y
            
        # max_sequence_length is depending on the length of most samples. see txet_length_stat() method for details
        self.max_sequence_length = text_length_stat(df_x,0.98)
        self.tokenizer = NNTokenPadding(params = {'max_sequence_length': self.max_sequence_length}, text_set=df_x)
        
        # transfer label set into one-hot sequence
        df_x ,word_index = self.tokenizer.extract(df_x)
        # transform label sets into one-hot sequence
        self.labelCategorizer = LabelCategorizer()
        self.labelCategorizer.fit_on_labels(df_y)
        df_y = self.labelCategorizer.to_category(df_y)
        
        df_train = df_x[:len(x)]
        df_train_label = df_y[:len(y)]
        df_val = df_x[len(x):]
        df_val_label = df_y[len(y):]
        
        # label_num is depending on the samples
        self.label_num = df_y.shape[1]
        
        # vocab_size is depending on the word_index, which is the return of NNTokenPadding(),extract().
        self.vocab_size = len(word_index) + 1
        
        #get given embedding_matrix from given w2v file
        embedding_matrix = None
        if self.use_external_embedding:
            assert self.w2v_fp is not None
            params ={
                'w2v_fp':self.w2v_fp
            }
            embedding_matrix = WordEmbedding(params).extract(word_index)
            self.embedding_dim = embedding_matrix.shape[1]



        #contruct computation gragh
        self.construct_graph(embedding_matrix)
        self.model.compile(loss = self.model_loss,
                           optimizer = self.model_optimizer,
                           metrics = self.model_metrics
                          )
        #add callback fuction for progress exposure and early-stopping
        #cb = ProgressExposure(int(len(df_train)/self.model_train_batchsize +1) * self.model_epoch, self.progress_fp)
        es = EarlyStopping(monitor='val_loss',patience = 0, verbose = 1, mode = 'min')
        
        #start training
        history = self.model.fit(df_train, df_train_label,
                                 validation_data = (df_val, df_val_label),
                                 epochs = self.model_epoch,
                                 batch_size = self.model_train_batchsize,
                                 #callbacks=[cb, es],verbose = 1)
                                 callbacks=[es],verbose = 1)
        
        self.train_report = history.history
        train_end_time = time.time()
        self.train_cost_time = train_end_time - train_start_time
        
        #printlog(self.getframework_name() + " " + self.get_model_name() + "" + "train process finished.")
        
    def predict(self, x):
        #printlog(self.get_framework_name() + "" + self.get_model_name() + "" + "predict process started")
        df_text, _ = self.tokenizer.extract(x)
        preds = self.model.predict(df_text, batch_size = self.model_test_batchsize, verbose = 1)
        preds = np.argmax(preds, axis = 1)
        preds = self.labelCategorizer.label_re_transform(preds)
        
        #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "predict process finished")
        return preds
    
    def score(self, x, y):
        #printlog(self.get_framwork_name() + "" + self.get_model_name() + "" + "score pocess started")
        df_x , _ = self.tokenizer.extract(x)
        df_label = self.labelCategorizer.to_category(y)
        
        #score[0] is loss, score[1] is acc
        scores = self.model.evaluate(df_x, df_label, verbose = 1)
        
        #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "score process finished.")
        return scores[1]
    
    def save(self, mp):
        #printlog(self.get_frameworks_name() + " " + self.get_model_name() + " " + "save process started.")
        self.model.save(mp + '.h5')
        pickle.dump(self.tokenizer, open(mp + '_tokenizer', 'wb'))
        self.labelCategorizer.save("./" + self.get_model_name() + '_label_map_relation.txt')
        #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "save process finished.")
        
    def load(self, mp):
        #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "load process started.")
        self.model = load_model(mp + '.h5')
        self.tokenizer = pickle.load(open(mp + '_tokenizer', 'rb'))
        self.labelCategorizer = LabelCategorizer()
        self.labelCategorizer.load('./' + self.get_model_name() + '_label_map_relation.txt')
        #printlog(self.get_framework_name() + " " + self.get_model_name() + " "  + "load process finished.")
        
    def get_default_args(self):
        params = {
            'use_external_embedding' : False,
            'embedding_trainable' : True,
            'embedding_dim' : 300,
            
            'dropout_rate' : 0.5,
            'filter_num' : 128,
            'filter_size' : [3,4,5],
            'conv_activation' : 'tanh',
            'conv_strides' : 1,
            'conv_padding' : 'valid',
            
            'pooling_sizes' : [3,4,5],
            'pooling_strides' : [3,4,5],
            'pooling_padding' : 'valid',
            
            'dense_size' : 256,
            'dense_activation' : 'tanh',
            
            'model_loss' : 'categorical_crossentropy',
            'model_optimizer' : 'Adadelta',
            'model_metrics' : ['accuracy'],
            'model_epoch':5,
            'model_train_batchsize':128,
            'model_test_batchsize':1024
        }
        return params
    
    def get_framework_name(self):
        return self.framework_name
    
    def get_model_name(self):
        return self.model_name
    
    def get_train_report(self):
        return self.train_cost_time, self.train_report

