# -*- coding: utf-8 -*-
import codecs
import os
import re
from collections import OrderedDict
import pandas as pd
import jieba as jb
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import multiprocessing
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation, Flatten
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Conv1D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
import time


def get_data(data_dir):
    training_X = []
    training_y = []
    testing_X = []
    testing_y = []

    for cur_category in range(1, 7):
        print '-'*100
        print cur_category
        training_input_file = codecs.open(filename=os.path.join(data_dir, 'sohu_corpus', 'cnn', 'training_' + str(cur_category) + '.cs'), mode='r', encoding='utf-8')
        testing_input_file = codecs.open(filename=os.path.join(data_dir, 'sohu_corpus', 'cnn', 'testing_' + str(cur_category) + '.cs'), mode='r', encoding='utf-8')
        for tmp_line in training_input_file:
            training_X.append(tmp_line)
            training_y.append(cur_category-1)
        for tmp_line in testing_input_file:
            testing_X.append(tmp_line)
            testing_y.append(cur_category-1)

    training_df = pd.DataFrame({'label' : training_y, 'text' : training_X})
    print 'training_df.shape:', training_df.shape

    testing_df = pd.DataFrame({'label' : testing_y, 'text' : testing_X})
    print 'testing_df.shape:', testing_df.shape
    return training_df, testing_df

def word2vec_train(combined_X, combined_y):
    model = Word2Vec(size=embedding_dim, min_count=n_exposures, window=window_size, workers=multiprocessing.cpu_count(), iter=n_iterations)
    model.build_vocab(combined_X)
    model.train(combined_X, total_examples=len(combined_X), epochs=model.iter)
    #model.save('./word_threshold_10/Word2vec_model.pkl')

    '''
    aa = model.wv.most_similar(u'新闻')
    for a in aa:
        print '-' * 10
        print a[0]
        print a[1]

    print 'type of combined_X:'
    print type(combined_X)
    '''

    index_dict, word_vectors, combined_X_ids = create_dictionary(model, combined_X)

    n_symbols = len(index_dict) + 1
    embedding_weights = np.zeros((n_symbols, embedding_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]

    #print 'type of combined_X_ids:', type(combined_X_ids)
    #print 'len of combined_X_ids:', len(combined_X_ids)


    X_train = combined_X_ids[0: 90000]
    X_test = combined_X_ids[90000:]
    y_train = combined_y[0: 90000]
    y_test = combined_y[90000:]

    '''
    r = random.random()
    random.shuffle(X_train, lambda : r)
    random.shuffle(y_train, lambda : r)

    rr = random.random()
    random.shuffle(X_test, lambda : rr)
    random.shuffle(y_test, lambda : rr)
    '''
    # X_train, X_test, y_train, y_test = train_test_split(combined_X_ids, combined_y, test_size=0.2)
    return n_symbols, embedding_weights, X_train, y_train, X_test, y_test

def word_to_index(src, w2indx):
    data = []
    cc = 0
    for sentence in src:
        '''
        if cc < 6:
            print '-' * 10
            print type(sentence)
            print sentence
        cc += 1
        '''
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)
        data.append(new_txt)
    return data

def create_dictionary(model, combined):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    '''
    cc = 0
    for k, v in gensim_dict.items():
        if cc < 10:
            print '-'*100
            print k
            print v
        else:
            break
        cc += 1
    '''
    word2index = {v: k+1 for k, v in gensim_dict.items()} # {word : index}

    word2vec = {word: model[word] for word in word2index.keys()}

    '''
    cc = 0
    for word in word2index.keys():
        if cc < 5:
            print '-'*100
            print word
            print model[word]
        else:
            break
        cc += 1
    '''

    word_index_combined = word_to_index(combined, word2index)
    '''
    for x in word_index_combined[0:5]:
        print x
    print 'after padding:'
    '''
    padded_data = sequence.pad_sequences(word_index_combined, maxlen=input_length)

    '''
    for x in padded_data[0:5]:
        print x
    '''

    return word2index, word2vec, padded_data


def cnn_train(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print '===================================cnn training with trainable=True'
    #print 'n_symbols:{}'.format(n_symbols)


    conv1 = Sequential([
        Embedding(input_dim=n_symbols, output_dim=embedding_dim, input_length=input_length, weights=[embedding_weights], trainable=True),
        Dropout(0.2),
        Conv1D(64, 5, padding='same', activation='relu'),
        Dropout(0.2),
        MaxPooling1D(),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.7),
        Dense(6, activation='softmax')]) # sigmoid
    conv1.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    #conv1.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=4, batch_size=64)
    conv1.fit(x_train, to_categorical(y_train), validation_data=(x_test, to_categorical(y_test)), epochs=4, batch_size=64)

def cnn_train_no_pretrained_vector_trigger(n_symbols, x_train, y_train, x_test, y_test):
    #print '====================================================cnn_train_no_pretrained_vector'
    #print 'n_symbols:{}'.format(n_symbols)

    conv1 = Sequential([
        Embedding(input_dim=n_symbols, output_dim=embedding_dim, input_length=input_length),
        Dropout(0.2),
        Conv1D(64, 5, padding='same', activation='relu'),
        Dropout(0.2),
        MaxPooling1D(),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.7),
        Dense(6, activation='softmax')]) # sigmoid
    conv1.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    #conv1.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=4, batch_size=64)
    conv1.fit(x_train, to_categorical(y_train), validation_data=(x_test, to_categorical(y_test)), epochs=1, batch_size=64)


def cnn_train_no_pretrained_vector(n_symbols, x_train, y_train, x_test, y_test):
    print '====================================================cnn_train_no_pretrained_vector'
    #print 'n_symbols:{}'.format(n_symbols)

    conv1 = Sequential([
        Embedding(input_dim=n_symbols, output_dim=embedding_dim, input_length=input_length),
        Dropout(0.2),
        Conv1D(64, 5, padding='same', activation='relu'),
        Dropout(0.2),
        MaxPooling1D(),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.7),
        Dense(6, activation='softmax')]) # sigmoid
    conv1.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    #conv1.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=4, batch_size=64)
    conv1.fit(x_train, to_categorical(y_train), validation_data=(x_test, to_categorical(y_test)), epochs=4, batch_size=64)
    end_time = time.time()
    print 'Time elapsed: {} minutes'.format((end_time-start_time) / 60)


def lstm_train(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    #print 'Defining a Simple Keras Model...'
    print 'lstm training'
    model = Sequential()
    model.add(Embedding(input_dim=n_symbols, # size of the vocabulary
                        output_dim=embedding_dim,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # length of input sequences
    model.add(LSTM(units=50, activation='relu')) # , inner_activation='hard_sigmoid'
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print 'Compiling the Model...'
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print "Train..."
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,verbose=1, validation_data=(x_test, y_test))

    print "Evaluate..."
    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    print 'Test score:', score

if __name__ == '__main__':

    #stopwordsCN = stopWords('./data/stopWords_cn.txt')
    #pre_process(u'我感冒了没药吃了')


    data_dirs = '/data/dse/lib/cassandra/nlp_related/text_classification'
    training_data, testing_data = get_data(data_dir=data_dirs)
    print 'training_data.shape:', training_data.shape
    print 'testing_data.shape:', testing_data.shape

    embedding_dim = 100
    #maxlen = 100
    n_iterations = 10
    n_exposures = 2
    window_size = 5
    batch_size = 32
    n_epoch = 4
    input_length = 300

    combined_src = np.concatenate((training_data['text'], testing_data['text']))
    combined_X = [x.split() for x in combined_src]
    combined_y = np.concatenate((training_data['label'], testing_data['label']))

    # lstm_train(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = word2vec_train(combined_X, combined_y)

    start_time = time.time()
    print 'start_time:', start_time
    cnn_train(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)

    #cnn_train_no_pretrained_vector_trigger(n_symbols, x_train, y_train, x_test, y_test)

    #cnn_train_no_pretrained_vector(n_symbols, x_train, y_train, x_test, y_test)


    '''
    for idx, row in training_data.iterrows():
        print '-'*10
        print row['label']
        print row['text'].encode('utf-8')


    print '='*100
    for idx, row in testing_data.iterrows():
        print '-' * 10
        print row['label']
        print row['text'].encode('utf-8')

    print 'load dict'
    jb.load_userdict('/data/dse/lib/cassandra/nlp_related/dict/jieba_dict_2017_05_12.txt')
    print 'load dict finished'
    '''



