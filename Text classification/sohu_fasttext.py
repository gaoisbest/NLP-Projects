# -*- coding:utf-8 -*-

import fasttext as ft
import os

file_path = './sohu_corpus/fasttext'


input_file = os.path.join(file_path, 'training.cs')
output = 'sohu_fasttext_06_30.model'
test_file = os.path.join(file_path, 'testing.cs')


dims = [64, 32, 100, 120, 200]
lrs = [0.5, 0.01, 0.05, 0.1, 0.2]
lr_update_rates = [100, 10, 20, 50, 200]
wss = [3, 5, 7, 10]
epoches = [5, 10, 20]
min_counts = [1, 2, 3]
negs = [2, 3, 5, 7]
losses = ['hs', 'ns', 'softmax']
word_ngramss = [1, 2, 3]
label_prefix = '__label__'

best_result = 0.0
best_paras = ''
for each_dim in dims:
    for each_lr in lrs:
        for each_lr_update_rate in lr_update_rates:
            for each_ws in wss:
                for each_epoch in epoches:
                    for each_min_count in min_counts:
                        for each_neg in negs:
                            for each_loss in losses:
                                #for each_word_ngram in word_ngramss:
                                curr_paras = 'dim:{}, lr:{}, lr_update_rate:{}, ws:{}, epoch:{}, min_count:{}, neg:{}, loss:{}'.format(each_dim, each_lr, each_lr_update_rate, each_ws, each_epoch, each_min_count, each_neg, each_loss)

                                print 'training:---------------------------'
                                print curr_paras

                                classifier = ft.supervised(input_file, output, dim=each_dim, lr=each_lr, ws=each_ws,
                                                           neg=each_neg, loss = each_loss, epoch=each_epoch, lr_update_rate=each_lr_update_rate,
                                                           min_count=each_min_count, label_prefix=label_prefix)
                                print 'testing---------------'
                                # Test the classifier
                                result = classifier.test(test_file)
                                print 'P@1:', result.precision
                                print 'R@1:', result.recall
                                if result.precision > best_result:
                                    best_result = result.precision
                                    best_paras = curr_paras

                                print '='*50+'best results:'
                                print best_result
                                print best_paras
                                print '='*50


'''

# best params
dim = 100
lr = 0.1 # 0.1
epoch = 10
min_count = 2
word_ngrams = 1
#bucket = 10000000 # 10000000  1000000
thread = 4
silent = 1
label_prefix = '__label__'
# bucket=bucket,
print 'training---------'
# Train the classifier
classifier = ft.supervised(input_file, output, dim=dim, lr=lr, epoch=epoch,
    min_count=min_count, word_ngrams=word_ngrams,
    thread=thread, silent=silent, label_prefix=label_prefix)

print 'testing---------------'
# Test the classifier
result = classifier.test(test_file)
print 'P@1:', result.precision
print 'R@1:', result.recall
print 'Number of examples:', result.nexamples
'''





#labels = classifier.predict_proba(texts, k=3)
#print labels
