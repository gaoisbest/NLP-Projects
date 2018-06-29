
# -*- coding: utf-8 -*-

import codecs
import os
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity
import re
import jieba as jie

class sohu_corpus(object):
    """Prepare the training data format
    """

    def __init__(self, fname, dic):
        self.fname = fname
        self.dic = dic

    def clean_text(self, text):
        # lower case
        text = text.strip().lower()
        # http:...
        text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)
        # @username
        # text = re.sub('@[^\s]+', '', text)

        text = re.sub(r'#[^#]+#', '', text)

        text = re.sub(u'\[\w+\]', '', text)

        text = re.sub(u'【\w+】', '', text)

        # text = re.sub(r'[0-9]+', '', text)

        text = re.sub(u'(回复|//)?@[\u4e00-\u9fa5a-zA-Z0-9_-]+(:|u"："|\s+)?', '', text)

        # split remove spaces
        text = ''.join(text.split())
        # white spaces
        text = re.sub(r'[\s]+', '', text)

        '''
            text = re.sub(
                u"([hH]ttp[s]{0,1})://[a-zA-Z0-9\.\-]+\.([a-zA-Z]{2,4})(:\d+)?(/[a-zA-Z0-9\-~!@#$%^&*+?:_/=<>.',;]*)?", '',
                text)  # remove http:xxx
            text = re.sub(u'#[^#]+#', '', text)  # remove #xxx#
            text = re.sub(u'回复@[\u4e00-\u9fa5a-zA-Z0-9_-]{1,30}:', '', text)  # remove "回复@xxx:"
            text = re.sub(u'@[\u4e00-\u9fa5a-zA-Z0-9_-]{1,30}', '', text)  # remove "@xxx"
            .lower()

            return text
        '''
        return jie.lcut(text)

    def __iter__(self):
        for line in codecs.open(filename=self.fname, encoding='utf-8', mode='r'):
            yield self.dic.doc2bow(self.clean_text(line), allow_update=True)


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
    training_file_path = './keywords/sentidata'

    # Lsi model
    dictionary = Dictionary()
    corpus = sohu_corpus(fname=os.path.join(training_file_path, 'neg_1.txt'), dic=dictionary)

    # save dictionary
    # dictionary.save(os.path.join(training_file_path, '07_11_dictionary.dict'))
    MmCorpus.serialize(os.path.join(training_file_path, '07_11_corpus_12.mm'), corpus)
    # dictionary = Dictionary.load(os.path.join(training_file_path, '07_11_dictionary.dict'))
    corpus_tfidf_mm = MmCorpus(os.path.join(training_file_path, '07_11_corpus_12.mm'))

    training_src_data = sogou_corpus_file(os.path.join(training_file_path, 'neg_1.txt'))
    training_src = []
    for each_file in training_src_data:
        training_src.append(each_file)

    # convert counts to tfidf
    tfidf = TfidfModel(corpus=corpus_tfidf_mm)

    index = MatrixSimilarity(tfidf[corpus_tfidf_mm])

    sims = index[tfidf[dictionary.doc2bow(['阳台', '打死'])]]
    print('doc2bow:')
    print(dictionary.doc2bow(['阳台']))
    print('tfidf:')
    print(tfidf[dictionary.doc2bow(['阳台'])])
    print(u'相似文档：\n')
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    for num, value in sims[:3]:
            print(str(num) + '\t' + str(value) + '\t' + training_src[num] + '\n')