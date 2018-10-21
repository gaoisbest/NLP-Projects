# -*- coding: utf-8 -*-

import codecs
import os
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel, LsiModel, LdaModel
from gensim.similarities import MatrixSimilarity

class sohu_corpus(object):
    """Prepare the training data format
    """
    def __init__(self, fname, dic):
        self.fname = fname
        self.dic = dic
    def __iter__(self):
        for line in codecs.open(filename=self.fname, encoding='utf-8', mode='r'):
            yield self.dic.doc2bow(line.split(), allow_update=True)

class sogou_corpus_file(object):
    """origin src file    
    """
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        for line in codecs.open(filename=self.fname, mode='r', encoding='utf-8'):
            yield line

if __name__ == '__main__':

    # training_file_path = 'E:/2017_Deep_learning/text similarity'
    training_file_path = './sogou_corpus'

    # Lsi model
    dictionary = Dictionary()
    corpus = sohu_corpus(fname=os.path.join(training_file_path, 'sohu_text_similarity_training.corpus'), dic=dictionary)

    # save dictionary
    #dictionary.save(os.path.join(training_file_path, '07_11_dictionary.dict'))
    MmCorpus.serialize(os.path.join(training_file_path, '07_11_corpus_12.mm'), corpus)
    #dictionary = Dictionary.load(os.path.join(training_file_path, '07_11_dictionary.dict'))
    corpus_tfidf_mm = MmCorpus(os.path.join(training_file_path, '07_11_corpus_12.mm'))

    # convert counts to tfidf
    tfidf = TfidfModel(corpus=corpus_tfidf_mm)
    corpus_tfidf = tfidf[corpus_tfidf_mm]

    # build the model
    lsi = LsiModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=200)
    index = MatrixSimilarity(lsi[corpus_tfidf])

    print 'b'
    training_src_data = sogou_corpus_file(os.path.join(training_file_path, 'sohu_text_similarity_training.corpus'))
    training_src = []
    for each_file in training_src_data:
        training_src.append(each_file)

    # read original test data
    final_results = codecs.open(os.path.join(training_file_path, 'lsi_similarity.res'), mode='a', encoding='utf-8')

    for test_doc in codecs.open(os.path.join(training_file_path, 'sohu_text_similarity_testing.corpus'), mode='r', encoding='utf-8'):
        test_doc_vec = tfidf[dictionary.doc2bow(test_doc.split())]
        test_doc_vec_lsi = lsi[test_doc_vec]
        final_results.write('='*50+'\n')
        final_results.write(u'输入文档：\n')
        final_results.write(test_doc+'\n')
        sims = index[test_doc_vec_lsi]
        final_results.write(u'相似文档：\n')
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        for num, value in sims[:3]:
            final_results.write(str(num)+'\t' + str(value) + '\t' + training_src[num]+'\n')
