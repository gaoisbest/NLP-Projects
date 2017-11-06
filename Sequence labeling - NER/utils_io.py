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

def full_to_half(s):
    """
    Convert full-width character to half-width one 
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)

def replace_html(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)

def input_from_line(line, char_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(' ', '$')
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id['<UNK>'] for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    return inputs


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


def result_to_json(string, tags):
    item = {'string': string, 'entities': []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == 'S':
            item['entities'].append({'word': char, 'start': idx, 'end': idx+1, 'type':tag[2:]})
        elif tag[0] == 'B':
            entity_name += char
            entity_start = idx
        elif tag[0] == 'I':
            entity_name += char
        elif tag[0] == 'E':
            entity_name += char
            item['entities'].append({'word': entity_name, 'start': entity_start, 'end': idx + 1, 'type': tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item

def save_config(config, config_file):
    """
    Save configuration of the model parameters for model deploy.
    Parameters are stored in json format.
    """
    with open(config_file, mode='w', encoding='utf8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def load_config(config_file):
    """
    Load configuration of the model for model deploy.    
    Parameters are stored in json format.
    """
    with open(config_file, encoding='utf8') as f:
        return json.load(f)

def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    Load word embedding from pre-trained file.
    """
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        # line format: [character_a, 0.0824, -0.335, ..., 'word_dim' embeddings]
        line = line.rstrip().split()
        if len(line) == word_dim + 1: # valid embedding
            # line[0] is a character
            pre_trained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)

    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[re.sub('\d', '0', word.lower())]
            c_zeros += 1
    print('Loaded %i pre-trained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with pretrained embeddings.' % (c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words))
    print('%i found directly, %i after lowercasing, %i after lowercasing + zero.' % (c_found, c_lower, c_zeros))
    return new_weights


def make_path(params):
    """
    Make folders for training and evaluation
    """
    if not os.path.isdir(params.summary_dir):
        os.makedirs(params.summary_dir)
    if not os.path.isdir(params.ckpt_dir):
        os.makedirs(params.ckpt_dir)


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by counts.
    """
    # x[0]: character
    # x[-1]: tag
    # tags is a list of list which only contain x[-1] (i.e., tag)
    tags = [[x[-1] for x in s] for s in sentences]
    dico = create_dictionary(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print('Found %i unique named entity tags.' % len(dico))
    return dico, tag_to_id, id_to_tag

def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of characters, sorted by counts.
    """
    # x[0]: character
    # x[1]: tag
    # chars is a list of list which only contain x[0] (i.e., character)
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dictionary(chars)
    # add two extra characters
    dico['<PAD>'] = 10000001 # padding symbol, 10000001 is counts
    dico['<UNK>'] = 10000000 # unknown symbol
    char_to_id, id_to_char = create_mapping(dico)
    print('Found %i unique words (%i in total).' % (len(dico), sum(len(x) for x in chars)))
    return dico, char_to_id, id_to_char

def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with chars that have a pre-trained embedding.  
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # 'pretrained' contains characters that have pre-trained embedding vector
    # line format: character_a 0.0824 -0.335 ... 100 embeddings
    pretrained = set([line.rstrip().split()[0].strip() for line in codecs.open(ext_emb_path, 'r', 'utf-8') if len(ext_emb_path) > 0])

    if chars is None: # add every character that has a pre-trained embedding to the dictionary
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else: # add the words that are given by `chars` (typically the words in the development and test sets.)
        for char in chars:
            if any(x in pretrained for x in [char, char.lower(), re.sub('\d', '0', char.lower())]) and char not in dictionary:
                dictionary[char] = 0

    char_to_id, id_to_char = create_mapping(dictionary)
    return dictionary, char_to_id, id_to_char

def create_dictionary(item_list):
    """
    Create the dictionary[character] = character counts.
    """
    assert type(item_list) is list
    dico = {}
    # items is a list, which represents the sentence, that contains each character, i.e., ['a', 'b', ...]
    for items in item_list:
        for item in items: # item is each character
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    # sort the dictionary in descending order
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    # assign id to each character, i.e., id:character
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    # character:id
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)

def load_sentences(path, zeros):
    """
    Read character-based input data. 
    [1] A line must contain at least a character and its tag.
    [2] Sentences are separated by empty line.
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num += 1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
            else:
                word= line.split()
            if len(word) >= 2:
                sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences

def clean(params, maps_file_name, config_file_name, results_file_path):
    """
    Clean current folder
    remove saved model and training log
    """
    if os.path.isfile(os.path.join(params.data_dir, maps_file_name)):
        os.remove(os.path.join(params.data_dir, maps_file_name))

    if os.path.isdir(params.ckpt_dir):
        shutil.rmtree(params.ckpt_dir)

    if os.path.isdir(params.summary_dir):
        shutil.rmtree(params.summary_dir)

    if os.path.isdir(os.path.join(params.data_dir, results_file_path)):
        shutil.rmtree(os.path.join(params.data_dir, results_file_path))

    if os.path.isdir('__pycache__'):
        shutil.rmtree('__pycache__')

    if os.path.isfile(os.path.join(params.data_dir, config_file_name)):
        os.remove(os.path.join(params.data_dir, config_file_name))