# -*- coding: utf-8 -*-
import jieba as jb
import codecs
import re
import os

def pre_process(text):
    """Word segmentation and filtering.
    """
    if text != None:
        newline_removed_list = []
        split_lines = text.splitlines()
        for each_line in split_lines:
            each_line = each_line.strip()
            each_line = re.sub(r'[\s+]', '', each_line)
            each_line = re.sub(u'(@[\u4e00-\u9fa5A-Za-z0-9_-]{1,30})', '', each_line)
            if len(each_line) > 0:
                newline_removed_list.append(re.sub(' +', '', each_line))

        filtered_str = ' '.join(newline_removed_list)
        if len(filtered_str) > 0:
            str_seg = [m.strip() for m in jb.lcut(filtered_str) if (m.strip() not in stopwordsCN) and (len(m.strip()) > 0)]
            if len(str_seg) >= 10:
                return ' '.join(str_seg)
            else:
                return 'failed to process'
        else:
            return 'failed to process'
    else:
        return 'failed to process'

def get_category(curr_http):
    rtn_category_num = '-1'
    for each_key, each_value in category_num_dict.iteritems():
        if curr_http.find(each_key) != -1:
            rtn_category_num = each_value
            break

    return rtn_category_num

def filter_article(src_sen):
    """Delete the advertisement (if have) in the article.
    """
    sentence_list = src_sen.splitlines()
    rtn_list = []
    for each_sentence in sentence_list:
        if (each_sentence.find(u'观看此文') == -1) and (each_sentence.find('ID') == -1) and (each_sentence.find(u'的用户会') == -1) and (each_sentence.find(u'内容来自网络') == -1) and (each_sentence.find(u'点击下方') == -1) and (each_sentence.find(u'点击下面') == -1) and (each_sentence.find(u'微信号') == -1) and (each_sentence.find(u'阅读原文') == -1) and (each_sentence.find(u'快来领取') == -1) and (each_sentence.find(u'公众号') == -1) and (each_sentence.find(u'转载自网络') == -1) and (each_sentence.find(u'小编有话说') == -1) and (each_sentence.find(u'长按') == -1) and (each_sentence.find(u'内容来源于网络') == -1) and (each_sentence.find(u'如有侵权') == -1) and (each_sentence.find(u'有人用微信聊天') == -1):
            rtn_list.append(each_sentence.strip())
        else:
            break
    return ' '.join(rtn_list)

def stopWords(stop_words_path):
    """Read the stop words. 
    """
    f = codecs.open(stop_words_path, encoding='utf-8')
    lines = []
    for line in f:
        line = line.rstrip('\n').rstrip('\r')
        lines.append(line)
    return lines

if __name__ == '__main__':

    #
    # cat news_sohusite_xml.dat | iconv -f gbk -t utf-8 -c > news_sohusite_xml.full.txt

    data_dir_path = 'E:/2017_Deep_learning/text classification/sougou_corpus'

    stopwordsCN = stopWords(os.path.join(data_dir_path, 'stopWords_cn_weixin.txt')) + [u'\ue40c']

    category_num_dict = {'auto.sohu.com': '1',
              'business.sohu.com': '2',
              'it.sohu.com': '3',
              'health.sohu.com': '4',
              'sports.sohu.com': '5',
              'travel.sohu.com': '6',
              'learning.sohu.com': '7',
              'career.sohu.com': '8',
              'cul.sohu.com': '9',
              'mil.news.sohu.com': '10',
              'house.sohu.com': '11',
              'yule.sohu.com': '12',
              'women.sohu.com': '13',
              'media.sohu.com': '14',
              'gongyi.sohu.com': '15'
              #'news.sohu.com': '16'
              }

    category_articles = {}
    cc = 0

    for dirpath, dirnames, filenames in os.walk(os.path.join(data_dir_path, 'news_sohusite_xml.full.txt')):
        for each_file in filenames:
            # naive for loop to parse file
            print os.path.join(dirpath, each_file)
            curr_file = os.path.join(dirpath, each_file)
            cur_src_corpus = codecs.open(curr_file, encoding='utf-8')

            curr_category = ''
            curr_text = ''

            for each_line in cur_src_corpus:

                if each_line.find('<url>') != -1:
                    cleaned_http = re.sub('<url>|</url>', '', each_line)
                    category_num = get_category(cleaned_http)
                    if category_num != '-1':
                        curr_category = category_num
                elif each_line.find('<content>') != -1:
                    cleaned_text = re.sub('<content>|</content>', '', each_line)
                    curr_text = cleaned_text
                if len(curr_category) > 0 and len(curr_text) > 0:
                    filtered_article = filter_article(curr_text)
                    processed = pre_process(filtered_article)
                    if processed != 'failed to process':
                        tmp_articles = category_articles.get(curr_category, [])
                        tmp_articles.append(processed)
                        category_articles[curr_category] = tmp_articles
                        cc += 1
                        print cc

                    curr_category = ''
                    curr_text = ''
                elif len(curr_text) > 0:
                    curr_category = ''
                    curr_text = ''


    print 'category_articles:{}'.format(len(category_articles))
    for curr_category_num, curr_articles in category_articles.iteritems():
        print '-'*10
        print curr_category_num
        print 'length:{}'.format(len(set(curr_articles)))
        curr_output = codecs.open(filename=os.path.join(data_dir_path, 'sohu_corpus_' + curr_category_num + '.corpus'), mode='a', encoding='utf-8')
        for each_article in set(curr_articles):
            curr_output.write(each_article + '\n')