# -*- coding:utf-8 -*-

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import codecs
import os

class sogou_corpus(object):
    """Prepare the training data format
    """
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        for idx, line in enumerate(codecs.open(filename=self.fname, mode='r', encoding='utf-8')):
            yield TaggedDocument(words=line.split(), tags=[idx])

class sogou_corpus_file(object):
    """origin src file    
    """
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        for line in codecs.open(filename=self.fname, mode='r', encoding='utf-8'):
            yield line

if __name__ == '__main__':

    #training_file_path = 'E:/2017_Deep_learning/text similarity'

    training_file_path = './sogou_corpus'
    training_file = sogou_corpus(os.path.join(training_file_path, 'sohu_text_similarity_training.corpus'))

    # build model
    model = Doc2Vec(documents=training_file, size=200, window=8, min_count=2, iter=40)

    # read original input
    training_src_data = sogou_corpus_file(os.path.join(training_file_path, 'sohu_text_similarity_training.corpus'))
    training_src = []
    for each_file in training_src_data:
        training_src.append(each_file)

    # read original test data
    final_results = codecs.open(os.path.join(training_file_path, 'doc2vec_similarity.res'), mode='a', encoding='utf-8')

    for test_doc in codecs.open(os.path.join(training_file_path, 'sohu_text_similarity_testing.corpus'), mode='r', encoding='utf-8'):
        test_doc_vec = model.infer_vector(test_doc.split())
        final_results.write('='*50+'\n')
        final_results.write(u'输入文档：\n')
        final_results.write(test_doc+'\n')
        sims = model.docvecs.most_similar([test_doc_vec], topn=3)
        final_results.write(u'相似文档：\n')
        for num, value in sims[:3]:
            final_results.write(str(num) + '\t' + str(value) + '\t' + training_src[num] + '\n')
