# -*- coding: utf-8 -*-

import os
import pickle
import itertools
import tensorflow as tf
import numpy as np
from collections import OrderedDict

from model import BrandsNERModel
from conlleval import return_report
from utils_train import BatchManager, prepare_dataset, update_tag_scheme
from utils_io import clean, make_path, load_sentences, augment_with_pretrained, tag_mapping, char_mapping, load_config, save_config, load_word2vec

flags = tf.app.flags

# input
flags.DEFINE_string('data_dir',          'data',   'Path for training, development, testing and embedding data.')

# output
flags.DEFINE_string('summary_dir',  'summaries',   'Path for training and testing summaries.')
flags.DEFINE_string('ckpt_dir',     'checkpoints',  'Path for saving model checkpoints.')

# pre-processing
flags.DEFINE_boolean('zeros',            True,       'Replace digits with zero.')
flags.DEFINE_boolean('lower',            True,       'Convert character to lower case.')
flags.DEFINE_string('tag_schema',     'iobes',    'Tagging schema iobes or iob')

# bi-directional lstm + crf model
flags.DEFINE_integer('word_dim',            20,        'Embedding dimension for word, 0 if not used.')
flags.DEFINE_integer('char_dim',           100,        'Embedding dimension for character.')
flags.DEFINE_integer('num_units',          100,        'Number of recurrent units in LSTM cell.')

# training
flags.DEFINE_float('learning_rate',       0.001,      'Initial learning rate.')
flags.DEFINE_float('max_gradient_norm',       5,      'Clip gradients to this norm.')
flags.DEFINE_float('batch_size',             20,      'Batch size to use during training.')
flags.DEFINE_float('keep_prop',             0.5,      'Initial dropout rate.')
flags.DEFINE_boolean('use_crf',            True,      'Use crf layer or softmax layer as the top layer.')
flags.DEFINE_integer('num_epoch',           100,      'Number of epochs.')

# util
flags.DEFINE_boolean('clean',              True,      'Clean all the training-related folders and files.')

FLAGS = flags.FLAGS

def create_model(session, Model_class, path, load_vec, config, id_to_char):
    """
    Train the new model or re-use trained model.
    """
    model = Model_class(config)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=path)
    if ckpt and ckpt.model_checkpoint_path:
        print('Reading model parameters from %s' % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print('Created model with fresh parameters.')
        session.run(tf.global_variables_initializer())
        # assign character embeddings
        emb_weights = session.run(model.char_embeddings.read_value())
        emb_weights = load_vec(config['character_embedding_file'], id_to_char, config['char_dim'], emb_weights)
        session.run(model.char_embeddings.assign(emb_weights))
    return model


def train(training_file_name, dev_file_name, test_file_name, maps_file_name, character_embedding_file_name, config_file_name):
    """
    Train main entrance.
    """
    training_file = os.path.join(FLAGS.data_dir, training_file_name)
    dev_file = os.path.join(FLAGS.data_dir, dev_file_name)
    test_file = os.path.join(FLAGS.data_dir, test_file_name)
    maps_file = os.path.join(FLAGS.data_dir, maps_file_name)
    embedding_file = os.path.join(FLAGS.data_dir, character_embedding_file_name)
    config_file = os.path.join(FLAGS.data_dir, config_file_name)

    # load data sets
    # brands.train, dev, test tagging schema: IOB
    # train_sentences format: [[['a', 'B-SHOE'], ['b', 'I-SHOE'], ['c', I-SHOE], ['d', 'O'], ...], [next training example data]]
    train_sentences = load_sentences(training_file, FLAGS.zeros)
    dev_sentences = load_sentences(dev_file, FLAGS.zeros)
    test_sentences = load_sentences(test_file, FLAGS.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    # train_sentences format: [[['a', 'B-SHOE'], ['b', 'I-SHOE'], ['c', E-SHOE], ['d', 'O'], ...], [next training example data]]
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)

    # create maps.pkl if not exist
    # maps.pkl contains: char_to_id, id_to_char, tag_to_id, id_to_tag
    if not os.path.isfile(maps_file):
        print('create map file')
        # create dictionary for each character
        dict_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
        # update dictionary by add the characters in embedding files or test data set
        dict_chars, char_to_id, id_to_char = augment_with_pretrained(dict_chars_train.copy(), embedding_file, list(itertools.chain.from_iterable([[w[0] for w in s] for s in dev_sentences])))
        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        # pickle data
        with open(maps_file, 'wb') as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        print('load map file')
        with open(maps_file, 'rb') as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # convert character, tag, word segmentation to id
    train_data = prepare_dataset(train_sentences, char_to_id, tag_to_id, FLAGS.lower)
    dev_data = prepare_dataset(dev_sentences, char_to_id, tag_to_id, FLAGS.lower)
    test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id, FLAGS.lower)
    print('%i / %i / %i sentences in train / dev / test.' % (len(train_data), len(dev_data), len(test_data)))

    # prepare mini-batch data
    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)

    # make path for store summary and model if not exist
    make_path(FLAGS)

    if os.path.isfile(config_file):
        config = load_config(config_file)
    else:
        config = OrderedDict()
        config['num_chars'] = len(char_to_id)
        config['char_dim'] = FLAGS.char_dim
        config['num_tags'] = len(tag_to_id)
        config['word_dim'] = FLAGS.word_dim
        config['num_units'] = FLAGS.num_units
        config['batch_size'] = FLAGS.batch_size
        config['character_embedding_file'] = os.path.join(FLAGS.data_dir, character_embedding_file_name)
        config['max_gradient_norm'] = FLAGS.max_gradient_norm
        config['keep_prop'] = FLAGS.keep_prop
        config['learning_rate'] = FLAGS.learning_rate
        config['zeros'] = FLAGS.zeros
        config['lower'] = FLAGS.lower
        config['use_crf'] = FLAGS.use_crf
        save_config(config, config_file)

    # config parameters for the tf.Session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # number of mini-batches per epoch
    steps_per_epoch = train_manager.len_data

    with tf.Session(config=tf_config) as sess:
        # train_writer = tf.summary.FileWriter(logdir=os.path.join(FLAGS.summary_dir, 'train'), graph=sess.graph)
        # test_writer = tf.summary.FileWriter(logdir=os.path.join(FLAGS.summary_dir, 'test'), graph=sess.graph)
        train_writer = tf.summary.FileWriter(logdir=FLAGS.summary_dir, graph=sess.graph)
        model = create_model(sess, BrandsNERModel, FLAGS.ckpt_dir, load_word2vec, config, id_to_char)
        loss = []
        for i in range(FLAGS.num_epoch):
            for mini_batch in train_manager.iter_batch(shuffle=True):
                global_step, mini_batch_cost, mini_batch_summary = model.step(sess, mini_batch, is_training=True, keep_prop=FLAGS.keep_prop)
                train_writer.add_summary(summary=mini_batch_summary, global_step=global_step)
                loss.append(mini_batch_cost)
                if global_step % 100 == 0:
                    print('iteration:{} step:{}/{}, NER loss:{:>9.6f}'.format(i+1, global_step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []
            # evaluate the model on development data
            best = evaluate(sess, model, 'dev', dev_manager, id_to_tag)
            # if have better dev F1 score until now, then save the model
            if best:
                model.saver.save(sess=sess, save_path=os.path.join(FLAGS.ckpt_dir, 'Brands_ner.ckpt'), global_step=model.global_step.eval())
            # report the test F1 score
            evaluate(sess, model, 'test', test_manager, id_to_tag)


def evaluate(sess, model, name, data, id_to_tag):
    print('====================== evaluate:{}'.format(name))

    # ner_results contains 'character - real tag - predicted tag' for all samples in 'data'
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.data_dir)

    for line in eval_lines:
        print(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == 'dev':
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            sess.run(model.best_dev_f1.assign(f1))
            print('new best dev f1 score:{:>.3f}'.format(f1))
        return f1 > best_test_f1
    elif name == 'test':
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            sess.run(model.best_test_f1.assign(f1))
            print('new best test f1 score:{:>.3f}'.format(f1))
        return f1 > best_test_f1

def test_ner(results, path):
    """
    Report the performance.
    """
    output_file = os.path.join(path, 'Brands_ner_predict.utf8')
    with open(output_file, 'w', encoding='utf-8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + '\n')
            to_write.append('\n')

        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines

def main(_):
    if FLAGS.clean:
        clean(FLAGS, 'maps.pkl', 'BrandsNERModel.config', 'Brands_ner_predict.utf8')
    train('brands.train', 'brands.dev', 'brands.test', 'maps.pkl', 'wiki_100.utf8', 'BrandsNERModel.config')

if __name__ == '__main__':
    tf.app.run(main)