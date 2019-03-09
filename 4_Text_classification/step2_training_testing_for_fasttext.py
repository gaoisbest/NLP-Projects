# -*- coding: utf-8 -*-
import jieba as jb
import codecs
import re
import os
import random

if __name__ == '__main__':

    data_dir_path = 'E:/2017_Deep_learning/text classification/sougou_corpus'

    considered_category = {
        'sohu_corpus_1.corpus': '1',
        'sohu_corpus_2.corpus': '2',
        'sohu_corpus_3.corpus': '3',
        'sohu_corpus_4.corpus': '4',
        'sohu_corpus_5.corpus': '5',
        'sohu_corpus_12.corpus': '6'
    }

    training_data_thres = 15000
    testing_data_thres = 3000

    curr_training_file = codecs.open(os.path.join(data_dir_path, 'news_sohusite_unique_for_fasttext_06_29', 'training.cs'), mode='a', encoding='utf-8')
    curr_testing_file = codecs.open(os.path.join(data_dir_path, 'news_sohusite_unique_for_fasttext_06_29', 'testing.cs'), mode='a', encoding='utf-8')

    for dirpath, dirnames, filenames in os.walk(os.path.join(data_dir_path, 'news_sohusite_unique_06_29')):
        # print all files in current dirpath
        for each_file in filenames:
            if each_file in considered_category.keys():
                print os.path.join(dirpath, each_file)
                curr_file = os.path.join(dirpath, each_file)
                cur_src_corpus = codecs.open(curr_file, mode='r', encoding='utf-8')
                curr_file_articles = []
                for each_line in cur_src_corpus:
                    curr_file_articles.append(each_line)

                random.shuffle(curr_file_articles)

                for each_train in curr_file_articles[0:training_data_thres]:
                    curr_training_file.write('__label__' + considered_category.get(each_file) + ' , ' + each_train)

                for each_test in curr_file_articles[training_data_thres:(training_data_thres+testing_data_thres)]:
                    curr_testing_file.write('__label__' + considered_category.get(each_file) + ' , ' + each_test)