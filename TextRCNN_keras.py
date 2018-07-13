
# coding: utf-8

# In[ ]:


from keras.layers import Input,Embedding,Flatten,Concatenate,Dense,Lambda,Dropout
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU, Bidirectional
from keras.backend import mean, max
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

class TextRCNN:
    def __init__(self, params):
        #printlogInit()

        self.model_name = 'RCNN'
        self.framework_name = 'keras'

        # w2v parameters
        self.default_w2v_fp = params.get('default_w2v_fp', None)
        self.w2v_fp = params.get('w2v_fp', 'default')
        if self.w2v_fp != None:
            if self.w2v_fp == 'defalut':
                self.w2v_fp = self.default_w2v_fp
            self.use_external_embedding = True
        else:
            self.use_external_embedding = False

        # layer parameters
        self.embedding_dim = int(params.get('embedding_dim', 300))
        self.use_external_embedding = bool(params.get('use_external_embedding', True))
        self.embedding_trainable = bool(params.get('embedding_trainable', True))

        self.dropout_rate = float(params.get('dropout_rate', 0.5))

        self.rnn_cell_type = params.get('rnn_cell_type', 'LSTM')
        self.rnn_cell_size = int(params.get('rnn_cell_size', 256))

        self.dense_size = int(params.get('dense_size', 256))
        self.dense_activation = params.get('dense_activation', 'tanh')

        # model parameters
        self.model_loss = params.get('loss', 'categorical_crossentropy')
        self.model_optimizer = params.get('optimizer', 'Adadelta')
        self.model_metrics= [params.get('metrics', 'accuracy')]
        self.model_epoch = int(params.get('epoch', 5))
        self.model_train_batchsize = int(params.get('train_batchsize', 128))
        self.model_test_batchsize = int(params.get('test_batchsize', 1024))

        #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "initialization finished.") 
    
    def construct_graph(self, embedding_matrix = None):
        # 1.set input layer
        input_layer = Input(shape=(self.max_sequence_length, ))
        
        # 2.set embedding layer with pretrained_embedding or not
        if self.use_external_embedding:
            assert self.use_external_embedding is not None
            embeddin_layer = Embedding(self.vocab_size,
                                       self.embedding_dim,
                                       mask_zero = True,
                                       weights = [embedding_matrix],
                                       input_lenght = self.max_sequence_length,
                                       trainable = self.embedding_trainable)(input_layer)
            
        else:
            embedding_layer = Embedding(self.vocab_size,
                                        self.embedding_dim,
                                        mask_zero = True,
                                        input_length = self.max_sequence_length,
                                        trainable = self.embedding_trainable)(input_layer)
        
        # 3 get bi-rnn layer
        bi_rnn_layer_left = LSTM(self.rnn_cell_size,return_sequences = True)(embedding_layer)
        bi_rnn_layer_right = LSTM(self.rnn_cell_size,return_sequences = True,go_backwards = True)(embedding_layer)
        if self.rnn_cell_type == 'GRU':
            bi_rnn_layer_left = GRU(self.rnn_cell_size,return_sequences = True)(embedding_layer)
            bi_rnn_layer_right = GRU(self.rnn_cell_size,return_sequences = True,go_backwards = True)(embedding_layer)
        elif self.rnn_cell_type == 'CuDNNLSTM':
            bi_rnn_layer_left = CuDNNLSTM(self.rnn_cell_size,return_sequences = True)(embedding_layer)
            bi_rnn_layer_right = CuDNNLSTM(self.rnn_cell_size,return_sequences = True,go_backwards = True)(embedding_layer)
        elif self.rnn_cell_type == 'CuDNNGRU':
            bi_rnn_layer_left = CuDNNGRU(self.rnn_cell_size,return_sequences = True)(embedding_layer)
            bi_rnn_layer_right = CuDNNGRU(self.rnn_cell_size,return_sequences = True,go_backwards = True)(embedding_layer)
        
        #to reverse the sequcence of the output of the go_backwards LSTM
        bi_rnn_layer_right = Lambda(lambda x: backend.reverse(x ,axes = 1))(bi_rnn_layer_right)
        
        # 4.combine bi-rnn's output(viewed as word's context) and embedding layer(word)
        concatenate_layer = Concatenate()([bi_rnn_layer_left,embedding_layer,bi_rnn_layer_right])

        # 5.add maxpool to extract the most significant feature
        MyMaxPool = Lambda(lambda x: max(x,axis = 1))
        maxpool = MyMaxPool(concatenate_layer)
        
        # 6.add dorpout layer 
        dropout_layer = Dropout(self.dropout_rate)(maxpool)
        
        # 7.add dense layer and output layer
        dense_layer = Dense(self.dense_size, activation = self.dense_activation)(dropout_layer)
        output_layer = Dense(self.label_num, activation = 'softmax')(dense_layer)
        
        self.model = Model(inputs = input_layer, outputs = output_layer)
         
        print(self.model.summary())
        #printlog(self.model.summary())
        
    def train(self, x, y, *, val_x = None, val_y=None):
        #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "train process started.")
        train_start_time = time.time()
        
        df_x = x
        df_y = y
        if val_x is not None and val_y is not None:
            df_x = x + df_x
            df_y = y + df_y
        
        # max_sequence_length is depending on the length of most samples.
        self.max_sequence_length = text_length_stat(df_x, 0.98)
        self.tokenizer = NNTokenPadding(params = {'max_sequence_length': self.max_sequence_length}, text_set = df_x)
        
        # transform texf dataset into sequence
        df_x, word_index = self.tokenizer.extract(df_x)
        
        # transform labels into one-hot sequence
        self.labelCategorizer = LabelCategorizer()
        self.labelCategorizer.fit_on_labels(df_y)
        df_y = self.labelCategorizer.to_category(df_y)
        
        df_train = df_x[:len(x)]
        df_train_label = df_y[:len(y)]
        df_val = df_x[len(x):]
        df_val_label = df_y[len(y):]
        
        # label_num is depending on the samples.
        self.label_num = df_y.shape[1]
        
        #vocab_size is depending on the word_index, which is the return of NNTOkenPadding.extract()
        self.vocab_size = len(word_index) + 1
        
        #get embedding_matrix from giveb w2v file
        embedding_matrix = None
        if self.use_external_embedding:
            assert self.w2v_fp is not None
            params = {
                'w2v_fp':self.w2v_fp
            }
            embedding_matrix = WordEmbedding(params).extract(word_index)
            self.embedding_dim = embedding_matrix.shape[1]
            
        # construct computation graph
        self.construct_graph(embedding_matrix)
        
        self.model.compile(loss = self.model_loss,
                           optimizer = self.model_optimizer,
                           metrics = self.model_metrics)
        
        #add callback function for progress exposure and early-stopping
        #es = EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='min')
        
        # start training
        history = self.model.fit(df_train,df_train_label,validation_data = (df_val, df_val_label),
                                 epochs = self.model_epoch,
                                 verbose = 1,
                                 batch_size = self.model_train_batchsize)
                                 #callbacks=[cb, es])
        self.train_report = history.history
        
        train_end_time = time.time()
        self.train_cost_time = train_end_time - train_start_time
        
        #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "train process finished.")
    
    def predict(self, x):
        #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "predict process started.")

        df_text, _ = self.tokenizer.extract(x)
        preds = self.model.predict(df_text, batch_size=self.model_test_batchsize, verbose=1)
        preds = np.argmax(preds, axis=1)
        preds = self.labelCategorizer.label_re_transform(preds)

        #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "predict process finished.")
        return preds

    def score(self, x, y):
        #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "score process started.")

        df_x, _ = self.tokenizer.extract(x)
        df_label = self.labelCategorizer.to_category(y)

        # scores[0] is loss, scores[1] is acc
        scores = self.model.evaluate(df_x, df_label, verbose=1)

        #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "score process finished.")
        return scores

    def save(self, mp):
        #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "save process started.")

        self.model.save(mp + '.h5')
        pickle.dump(self.tokenizer, open(mp + '_tokenizer', 'wb'))
        self.labelCategorizer.save('./' + self.get_model_name() + '_label_map_relation.txt')

        #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "save process finished.")

    def load(self, mp):
        #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "load process started.")

        self.model = load_model(mp + '.h5')
        self.tokenizer = pickle.load(open(mp + '_tokenizer', 'rb'))
        self.labelCategorizer = LabelCategorizer()
        self.labelCategorizer.load('./' + self.get_model_name() + '_label_map_relation.txt')

       #printlog(self.get_framework_name() + " " + self.get_model_name() + " " + "load process finished.") 
        
    def get_default_args(self):

        params = {
            'use_external_embedding': False,
            'embedding_dim': 300,
            'embedding_trainable': True,

            'dropout_rate': 0.5,

            'rnn_cell_type': 'LSTM',
            'rnn_cell_size': 256,

            'dense_size': 256,
            'dense_activation': 'tanh',

            'model_loss': 'categorical_crossentropy',
            'model_optimizer': 'Adadelta',
            'model_metrics': ['accuracy'],
            'model_epoch': 5,
            'model_train_batchsize': 128,
            'model_test_batchsize': 1024
        }
        return params     
    
    def get_framework_name(self):
        return self.framework_name

    def get_model_name(self):
        return self.model_name

    def get_train_report(self):
        return self.train_cost_time, self.train_report      

