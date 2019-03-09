# -*- coding: utf-8 -*-
import os
import codecs
from gensim.models import Word2Vec

class SogouCorpusFile(object):
    """read corpus file
    """
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        for line in codecs.open(filename=self.fname, mode='r', encoding='utf-8'):
            yield line.split()

training_file_path = os.path.join('E:/2017_Deep_learning/word2vec', 'word_vector_108000.cs')
#training_file_path = os.path.join('.', 'word_vector_108000.cs')

training_file = SogouCorpusFile(training_file_path)


# model = Word2Vec(sentences=training_file, min_count=1)
# has the same function as the following three line codes
model = Word2Vec(min_count=1, size=128, window=5, sg=1, negative=5, sample=0.001, iter=5) # initialize an empty model
model.build_vocab(sentences=training_file)
model.train(sentences=training_file, total_examples=model.corpus_count, epochs=model.iter)

# save the model
# model.save(save_path)

# and load the model
#model = Word2Vec.load(save_path)


aa = model.most_similar(u'奔驰')[0:10]
for word, score in aa:
    print('{} {}'.format(word.encode('utf-8'), score))
'''
Results:
宝马 0.853861570358
奥迪 0.840906202793
梅赛德斯 0.831308066845
雷克萨斯 0.762128651142
英菲尼迪 0.756845831871
沃尔沃 0.756216704845
路虎 0.755097270012
捷豹 0.741012334824
丰田 0.739406943321
豪华车 0.713670134544
'''






'''
# NOTE: compute_loss is not supported when I install the latest version of gensim
# simple grid search for parameter tuning
dims = [64, 128, 256]
min_counts = [3, 5, 7]
sgs = [0, 1]
negative = [5, 7, 10]
iters = [5, 10, 20]

best_score = sys.maxint

for each_dim in dims:
    for each_min_count in min_counts:
        for each_iter in iters:
            for each_sg in sgs:
                if each_sg == 1: # skip-gram model is used
                    for each_neg in negative:
                        model_with_loss = Word2Vec(sentences=training_file, size=each_dim, min_count=each_min_count, iter=each_iter, compute_loss=True, sg=1, negative=each_neg)
                        training_loss = model_with_loss.get_latest_training_loss()
                        if training_loss < best_score:
                            print 'best parameters: size:{}, min_count:{}, iter:{}, sg:{}, negative:{}, loss:{}'.format(each_dim, each_min_count, each_iter, each_sg, each_neg, training_loss)
                        else:
                            print 'best parameters: size:{}, min_count:{}, iter:{}, sg:{}, negative:{}, loss:{}'.format(each_dim, each_min_count, each_iter, each_sg, each_neg, training_loss)
                else: # hierarchical softmax model is used
                    model_with_loss = Word2Vec(sentences=training_file, size=each_dim, min_count=each_min_count, iter=each_iter, compute_loss=True, sg=0)
                    training_loss = model_with_loss.get_latest_training_loss()
                    if training_loss < best_score:
                        print 'best parameters: size:{}, min_count:{}, iter:{}, sg:{}, negative:{}, loss:{}'.format(each_dim, each_min_count, each_iter, each_sg, each_neg, training_loss)
                    else:
                        print 'current parameters: size:{}, min_count:{}, iter:{}, sg:{}, negative:{}, loss:{}'.format(each_dim, each_min_count, each_iter, each_sg, each_neg, training_loss)
'''


