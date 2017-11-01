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

# see https://radimrehurek.com/gensim/models/word2vec.html for detailed parameter explanation
# model = Word2Vec(sentences=training_file, min_count=1)
# has the same function as the following three line codes
model = Word2Vec(min_count=1, size=128, window=5, sg=0, hs=1, cbow_mean=1, sample=0.001, iter=5) # initialize an empty model
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
奥迪 0.622578263283
宝马 0.604659199715
英菲尼迪 0.569936037064
雷克萨斯 0.548489332199
豪华车 0.521981656551
捷豹 0.518993020058
豪车 0.517318427563
轿车 0.513283610344
法拉利 0.511121630669
沃尔沃 0.507730364799
'''
