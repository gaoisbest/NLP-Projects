# -*- coding: utf-8 -*-
import os
import codecs
from gensim.models import Word2Vec
import numpy as np
from scipy.spatial.distance import cosine

class sogou_corpus(object):
    """Prepare the training data format
    """
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        for line in codecs.open(filename=self.fname, mode='r', encoding='utf-8'):
            yield line.split()


def word_vec_averaging(w2v, words, dim):
    """
    Average all words vectors in one sentence.
    :param w2v: model
    :param words: input sentence
    :param dim: 'size' of model
    :return: the averaged word vectors as the vector for the sentence
    """
    vec_mean = np.array((dim,), dtype=np.float32)
    word_num = 0
    for word in words:
        if word in w2v.vocab:
            word_num += 1
            vec_mean = np.add(vec_mean, w2v[word])
    if word_num > 0:
        vec_mean = np.divide(vec_mean, word_num)
    return vec_mean


if __name__ == '__main__':

    # training_file_path = 'E:/2017_Deep_learning/text similarity'

    training_file_path = './sogou_corpus'
    training_file = sogou_corpus(os.path.join(training_file_path, 'sohu_text_similarity_training.corpus'))

    # build model
    ndim = 200
    model = Word2Vec(sentences=training_file, size=ndim, min_count=1)
    #vec = model[u'彭于晏']
    #print vec
    #print vec.shape

    # model[]
    #model.wv['computer']
    training_vecs = []
    training_src = []
    for each_file in training_file:
        training_vecs.append(word_vec_averaging(w2v=model, words=each_file, dim=ndim))
        training_src.append(' '.join(each_file))

    # read original test data
    final_results = codecs.open(os.path.join(training_file_path, 'word2vec_mean_similarity.res'), mode='a', encoding='utf-8')

    for test_doc in codecs.open(os.path.join(training_file_path, 'sohu_text_similarity_testing.corpus'), mode='r', encoding='utf-8'):
        test_doc_vec = word_vec_averaging(w2v=model, words=test_doc.split(), dim=ndim)
        # print test_doc_vec
        final_results.write('='*50+'\n')
        final_results.write(u'输入文档：\n')
        final_results.write(test_doc+'\n')
        final_results.write(u'相似文档：\n')
        sims = []
        for each_train_vec in training_vecs:
            sims.append(1-cosine(each_train_vec, test_doc_vec))

        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        for num, value in sims[:3]:
            final_results.write(str(num) + '\t' + str(value) + '\t' + training_src[num] + '\n')
