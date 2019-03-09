# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import pickle

from utils_io import load_config, input_from_line
from model import BrandsNERModel

def create_model(session, Model_class, path, config):
    # create model
    model = Model_class(config)
    # reuse parameters if exists
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reading model parameters from %s' % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No model found!')
    return model


def test(maps_file_name, config_file_name, model_path):
    config = load_config(config_file_name)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with open(maps_file_name, 'rb') as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, BrandsNERModel, model_path, config)
        while True:
            line = input('请输入测试句子:')
            result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            print(result)

if __name__ == '__main__':
    maps_file_name = os.path.join('data', 'maps.pkl')
    config_file_name = os.path.join('data', 'BrandsNERModel.config')
    model_path = 'checkpoints'
    test(maps_file_name, config_file_name, model_path)