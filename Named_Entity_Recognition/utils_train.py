# -*- coding: utf-8 -*-

import os
import re
import codecs
import shutil
import jieba as jie
import random
import math
import logging
import numpy as np
import json

def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to tag_scheme.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')

def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True

def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags

def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

class BatchManager(object):
    def __init__(self, data,  batch_size):
        # data format: [[string, chars, segs, tags], [next sentence information]]
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        #print('len(data):{}, batch_size:{}, num_batch:{}'.format(len(data), batch_size, num_batch))
        # for minimal padding, sort them according to sequence length
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch): # iterate the data
            batch_data.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, char, seg, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)
        '''
        take a look at first two training examples.
        
        print(strings[0:2])
        [['一', '亿', '二', '的', '代', '价', '（', '中', '华', '环', '保', '世', '纪', '行', '）'], ['钱', '其', '琛', '会', '见', '香', '港', '泉', '州', '同', '乡', '会', '访', '问', '团']]
        
        print(chars[0:2])
        [[7, 523, 283, 3, 122, 417, 306, 10, 244, 393, 138, 142, 374, 32, 307], [694, 147, 1966, 20, 319, 555, 514, 1283, 461, 66, 406, 20, 405, 123, 253]]
        
        print(segs[0:2])
        [[1, 3, 0, 0, 1, 3, 0, 1, 3, 1, 3, 1, 3, 0, 0], [1, 2, 3, 1, 3, 1, 3, 1, 3, 1, 2, 3, 1, 2, 3]]
        
        print(targets[0:2])
        [[0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0], [8, 7, 9, 0, 0, 4, 1, 1, 1, 1, 1, 1, 1, 1, 5]]
        '''
        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists and each inner list contains:
    - string: character list
    - chars: character id list
    - segs: word segmentation encoding, see get_seg_features for more detail
    - tags: tag id list
    """

    none_index = tag_to_id['O']

    def f(x):
        return x.lower() if lower else x

    data = []
    # sentences is a list of list
    for s in sentences: # s is a list
        string = [w[0] for w in s]
        # convert each char to its id
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>'] for w in string]
        segs = get_seg_features("".join(string))
        if train:
            # convert each tag to its id
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags])

    return data

def get_seg_features(string):
    """
    Chinese word segmentation with jieba.
    Features are represented in BIES format, i.e., B:1, I:2, E:3, S:0.
    
    For example:
    string: u'我买了富士康手机'
    encoding: [0, 0, 0, 1, 2, 3, 1, 3]    
    """
    seg_feature = []

    for word in jie.cut(string):
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)

