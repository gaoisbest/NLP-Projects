# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from utils_io import result_to_json
from utils_train import iobes_iob


class BrandsNERModel:
    """
    Model for recognizing brands.    
    """

    def __init__(self, config):
        self.learning_rate = config['learning_rate']
        self.char_dims = config['char_dim']
        self.num_chars = config['num_chars']
        self.word_dims = config['word_dim']
        self.num_words_types = 4 # BIES
        self.num_tags = config['num_tags']
        self.rnn_units = config['num_units']
        self.max_gradient_norm = config['max_gradient_norm']

        # input placeholders
        self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='char_inputs')
        self.word_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='word_inputs')
        self.tags = tf.placeholder(dtype=tf.int32, shape=[None, None], name='brand_tags')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

        self.batch_size = tf.shape(input=self.char_inputs)[0]
        self.num_steps = tf.shape(input=self.char_inputs)[1]

        self.global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False, name='global_step')
        # model's best development F1 score
        self.best_dev_f1 = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=False, name='best_dev_f1')
        # model's best test F1 score
        self.best_test_f1 = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=False, name='best_test_f1')

        # use crf or not
        self.use_crf = config['use_crf']

        # model architecture
        # embedding layer
        char_word_embeddings = self.embedding_layer(self.char_inputs, self.word_inputs)

        # dropout
        # according to 'Neural Architectures for Named Entity Recognition' Section 4.3
        rnn_inputs = tf.nn.dropout(x=char_word_embeddings, keep_prob=self.keep_prob, name='lstm_inputs_dropout')

        # get the actual sequence length of this batch
        self.seq_lengths = tf.cast(tf.reduce_sum(input_tensor=tf.sign(tf.abs(self.char_inputs)), axis=1), tf.int32)

        # bi-directional rnn layer
        rnn_outputs = self.birnn_layer(rnn_inputs, self.rnn_units, self.seq_lengths)

        self.logits = self.projection_layer(rnn_outputs)

        self.cost = self.cost_layer(self.logits, self.seq_lengths, self.use_crf)

        self.train_op = self.optimize(self.cost)

        self.saver = tf.train.Saver(tf.global_variables())

    def embedding_layer(self, char_inputs, word_inputs):
        """
        Character and word segmentation embedding.
        :return: concatenated character and word segmentation embedding.
        """
        char_and_word_embeddings = []
        with tf.variable_scope('char_embedding_layer'), tf.device('/cpu:0'):
            self.char_embeddings = tf.get_variable(name='char_embeddings', shape=[self.num_chars, self.char_dims], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            char_and_word_embeddings.append(tf.nn.embedding_lookup(params=self.char_embeddings, ids=char_inputs, name='char_embeddings_lookup'))
        if self.word_dims > 0:
            with tf.variable_scope('word_embedding_layer'), tf.device('/cpu:0'):
                self.word_embeddings = tf.get_variable(name='word_embeddings', shape=[self.num_words_types, self.word_dims], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                char_and_word_embeddings.append(tf.nn.embedding_lookup(params=self.word_embeddings, ids=word_inputs, name='word_embeddings_loopup'))
        # shape of rtn_embeddings: [None, None, self.char_dims + self.word_dims]
        # axis=2 is also ok
        rtn_embeddings = tf.concat(values=char_and_word_embeddings, axis=-1)
        return rtn_embeddings

    def birnn_layer(self, rnn_inputs, rnn_num_units, seq_lengths):
        """
        Bi-directional LSTM model.
        :return: concatenated forward and backward outputs.
        """
        with tf.variable_scope('birnn_layer'):
            rnn_cells = {}
            for tmp in ['forward', 'backward']:
                # according to 'Neural Architectures for Named Entity Recognition' Section 2.1
                # CoupledInputForgetGateLSTMCell from paper 'LSTM: A Search Space Odyssey', where f = 1- i.
                rnn_cells[tmp] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(num_units=rnn_num_units, use_peepholes=True, initializer=tf.contrib.layers.xavier_initializer())
            # obtain 'contextual word representation' through bi-rnn according to https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_cells['forward'], cell_bw=rnn_cells['backward'], inputs=rnn_inputs, sequence_length=seq_lengths, dtype=tf.float32)
        # concatenate forward output and backward output: [None, None, self.rnn_units + self.rnn_units]
        return tf.concat(values=outputs, axis=-1)

    def projection_layer(self, rnn_outputs):
        """
        Two hidden fully connected layers for computing tag scores for each character.
        :param rnn_outputs: the outputs of birnn_layer 
        :return: tag scores matrix P 
        """
        with tf.variable_scope('projections'):
            project_W = tf.get_variable(name='project_W', shape=[2 * self.rnn_units, self.rnn_units], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            # use tf.zeros_initializer()
            project_b = tf.get_variable(name='project_b', shape=[self.rnn_units], dtype=tf.float32, initializer=tf.zeros_initializer())
            rnn_outputs_flat = tf.reshape(tensor=rnn_outputs, shape=[-1, 2 * self.rnn_units])
            project_Z = tf.nn.xw_plus_b(x=rnn_outputs_flat, weights=project_W, biases=project_b, name='projection_Z')
            # activation
            project_A = tf.tanh(project_Z, name='projection_A')
        with tf.variable_scope('logits'):
            logits_W = tf.get_variable(name='logits_W', shape=[self.rnn_units, self.num_tags], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            logits_b = tf.get_variable(name='logits_b', shape=[self.num_tags], dtype=tf.float32, initializer=tf.zeros_initializer())
            logits = tf.nn.xw_plus_b(x=project_A, weights=logits_W, biases=logits_b, name='matrix_P')
            logits_format = tf.reshape(tensor=logits, shape=[-1, self.num_steps, self.num_tags])

        # I have tried only one single hidden layer, but the F1 score is lower about 0.3%, the codes likes:
        '''
        with tf.variable_scope('logits'):
            logits_W = tf.get_variable(name='logits_W', shape=[2 * self.rnn_units, self.num_tags], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            logits_b = tf.get_variable(name='logits_b', shape=[self.num_tags], dtype=tf.float32, initializer=tf.zeros_initializer())
            rnn_outputs_flat = tf.reshape(tensor=rnn_outputs, shape=[-1, 2 * self.rnn_units])
            logits = tf.nn.xw_plus_b(x=rnn_outputs_flat, weights=logits_W, biases=logits_b, name='matrix_P')
            logits_format = tf.reshape(tensor=logits, shape=[-1, self.num_steps, self.num_tags])
        '''
        return logits_format

    def cost_layer(self, logits, seq_lengths, use_crf):
        """
        Model cost for crf layer and softmax layer.
        """
        if use_crf:
            with tf.variable_scope('crf_cost'):
                # reference codes see https://github.com/glample/tagger/blob/master/model.py line 288
                small = -1000.0
                # let y_-1 (i.e., self.num_tags) and y_n(i.e., self.num_tag+1) be the start and end tags of the sequence, respectively.
                # shape of seq_start_logits: [self.batch_size, 1, self.num_tags + 2], here 2 means the start and end tags
                # let seq_0 be the added start character of the sequence, then the ner tag of seq_0 is y_-1, since the second last 0 (tf.zeros()) > -1000.0 (small * tf.ones())
                seq_start_logits = tf.concat(values=[small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1]), small * tf.ones(shape=[self.batch_size, 1, 1])], axis=-1)

                # shape of seq_end_logits: [self.batch_size, 1, self.num_tags+2]
                # let seq_n be the added end character of the sequence, then the ner tag of seq_n is y_n, since the last 0 (tf.zeros()) > -1000.0 (small * tf.ones())
                #seq_end_logits = tf.concat(values=[small * tf.ones(shape=[self.batch_size, 1, self.num_tags + 1]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
                seq_end_logits = tf.concat(values=[small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), small * tf.ones(shape=[self.batch_size, 1, 1]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)

                # padding the logits with small values, small values make sure that padding will not affect acutal predicted values.
                # shape of padded_logits: [self.batch_size, self.num_steps, self.num_tags + 2]
                padded_logits = tf.concat(values=[logits, tf.cast(small * tf.ones([self.batch_size, self.num_steps, 2]), tf.float32)], axis=-1)

                # shape of final_logits: [self.batch_size, self.num_steps + 2, self.num_tags + 2]
                final_logits = tf.concat(values=[seq_start_logits, padded_logits, seq_end_logits], axis=1)

                # padding the actual tags
                # the ner tag of padded seq_0 and seq_n is self.num_tags and self.num_tags+1, respectively
                # shape of padded final_tags: [self.batch_size, self.num_steps + 2]
                seq_start_tags = tf.cast(self.num_tags * tf.ones(shape=[self.batch_size, 1]), tf.int32)
                seq_end_tags = tf.cast((self.num_tags+1) * tf.ones(shape=[self.batch_size, 1]), tf.int32)
                final_tags = tf.concat(values=[seq_start_tags, self.tags, seq_end_tags], axis=-1)

                log_likelihood, self.transition_matrix = tf.contrib.crf.crf_log_likelihood(inputs=final_logits, tag_indices=final_tags, sequence_lengths=seq_lengths+2)
                crf_cost = tf.reduce_mean(-log_likelihood)

                tf.summary.scalar(name='cost', tensor=crf_cost)
                tf.summary.histogram(name='histogram_cost', values=crf_cost)
                self.summary_op = tf.summary.merge_all()
                return crf_cost
        else:
            with tf.variable_scope('softmax_cost'):
                cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tags, logits=logits, name='cross_entropy_cost')
                # only compute seq_lengths cost
                masked_seq = tf.sequence_mask(lengths=seq_lengths)
                cross_entropy_loss = tf.boolean_mask(tensor=cross_entropy_loss, mask=masked_seq)
                softmax_cost = tf.reduce_mean(cross_entropy_loss)

                tf.summary.scalar(name='cost', tensor=softmax_cost)
                tf.summary.histogram(name='histogram_cost', values=softmax_cost)
                self.summary_op = tf.summary.merge_all()
                return softmax_cost

    def optimize(self, cost):
        """
        Adam optimizer with gradient clipping. 
        """
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # compute_gradients return: [(gradient_a, variable_a), (gradient_b, variable_b)]
            # gradients: (gradient_a, gradient_b)
            # variables: (variable_a, variable_b)
            gradients, variables = zip(*optimizer.compute_gradients(loss=cost))
            clipped_gradients, _ = tf.clip_by_global_norm(t_list=gradients, clip_norm=self.max_gradient_norm)
            optimize = optimizer.apply_gradients(grads_and_vars=zip(clipped_gradients, variables), global_step=self.global_step)
        return optimize
    

    def step(self, sess, mini_batch, is_training, keep_prop):
        """
        Run the model one time.
        :param sess: tensorflow session.
        :param mini_batch: mini_batch data.
        :param is_training: flag denotes whether training step or testing step.
        :param keep_prop: keep probability of dropout.
        :return: statistics of this mini-batch.
        """
        _, tmp_chars, tmp_words, tmp_tags = mini_batch

        mini_batch_fd = {}
        mini_batch_fd[self.char_inputs] = np.asarray(tmp_chars)
        mini_batch_fd[self.word_inputs] = np.asarray(tmp_words)
        mini_batch_fd[self.keep_prob] = keep_prop

        if is_training:
            mini_batch_fd[self.tags] = np.asarray(tmp_tags)
            _, mini_batch_global_step, mini_batch_cost, mini_batch_summary = sess.run([self.train_op, self.global_step, self.cost, self.summary_op], feed_dict=mini_batch_fd)
            return mini_batch_global_step, mini_batch_cost, mini_batch_summary
        else:
            seq_length, predictions = sess.run([self.seq_lengths, self.logits], feed_dict=mini_batch_fd)
            return seq_length, predictions

    def decode(self, logits, seq_length, transition_matrix):
        """
        Decode the best tags via Viterbi algorithm.
        :param logits: predicted tag scores.
        :param seq_length: actual sequence lengths.
        :param transition_matrix: tag transition matrix.
        :return: best tags for each character.
        """
        best_tags = []
        small = -1000.0
        start_logits = np.asarray(a=[[small] * self.num_tags + [0, small]])
        end_logits = np.asarray(a=[[small] * self.num_tags + [small, 0]])
        # iterate each sequence
        for tmp_logits, tmp_length in zip(logits, seq_length):
            tmp_logits = tmp_logits[:tmp_length]
            padded_logits = np.concatenate([tmp_logits, small * np.ones(shape=[tmp_length, 2])], axis=1)
            # final_logits: [seq_len+2, num_tags+2]
            final_logits = np.concatenate([start_logits, padded_logits, end_logits], axis=0)
            # score: [seq_len, num_tags] matrix
            best_tag, _ = tf.contrib.crf.viterbi_decode(score=final_logits, transition_params=transition_matrix)
            # 1:len(best_tag)-1 means excludes start and end tags
            best_tags.append(best_tag[1:len(best_tag)-1])
        return best_tags


    def evaluate(self, sess, data_manager, id_to_tag):
        """
        Evaluate the model performance on dev or test data set.
        :param sess: tensorflow session.
        :param data_manager: dev or test data manager.
        :param id_to_tag: convert tag id to tag token.
        :return: [character_1 - real tag_1 - predicted tag_1, character_2 - real tag_2 - predicted tag_2, ...]
        """
        results = []
        if self.use_crf:
            transition_matrix = sess.run(self.transition_matrix)
        
        for mini_batch in data_manager.iter_batch():
            tmp_strings = mini_batch[0]
            tmp_tags = mini_batch[-1]
            tmp_lengths, tmp_logits = self.step(sess, mini_batch, is_training=False, keep_prop=1.0)
            
            if self.use_crf:
                batch_paths = self.decode(tmp_logits, tmp_lengths, transition_matrix)
            else:
                batch_paths = sess.run(tf.cast(tf.argmax(tmp_logits, axis=-1), tf.int32))
                
            for i in range(len(tmp_strings)):
                result = []
                string = tmp_strings[i][:tmp_lengths[i]]
                # real tags
                gold = iobes_iob([id_to_tag[int(x)] for x in tmp_tags[i][:tmp_lengths[i]]])
                # predicted tags
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:tmp_lengths[i]]])
                # for each sample in one batch, store the character, real tag and predicted tag
                for char, gold, pred in zip(string, gold, pred):
                    result.append(' '.join([char, gold, pred]))
                # stores the whole data for all mini-batches
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        lengths, scores = self.step(sess, inputs, is_training=False, keep_prop=1.0)
        
        if self.use_crf:
            transition_matrix = sess.run(self.transition_matrix)
            batch_paths = self.decode(scores, lengths, transition_matrix)
        else:
            batch_paths = sess.run(tf.cast(tf.argmax(scores, axis=-1), tf.int32))
            
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)
