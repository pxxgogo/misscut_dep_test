# -*- coding: utf-8 -*-


import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def data_type():
    return  tf.float32



class RNN(object):
    """The PTB model."""

    def __init__(self, config):
        self._batch_size = config['batch_size']
        self._hidden_size = config['hidden_size']
        self._word_vocab_size = config['word_vocab_size']
        self._label_vocab_size = config['label_vocab_size']
        self._config = config
        self._data_placeholder = tf.placeholder(tf.int32, [None, self._config["sequence_length"]])
        self._cost_op = self.calculate_cost()

    def calculate_cost(self):
        input_data = self._data_placeholder
        batch_size = tf.shape(input_data)[0]
        word_rnn_cell_list = []
        for nn_info in range(self._config['layer_num']):
            # rnn_cell = tf.contrib.rnn.BasicGRUCell(self._hidden_size, forget_bias=0.0, state_is_tuple=True)
            rnn_cell = tf.contrib.rnn.GRUCell(self._hidden_size)

            word_rnn_cell_list.append(rnn_cell)

        word_cell = tf.contrib.rnn.MultiRNNCell(word_rnn_cell_list, state_is_tuple=True)

        label_rnn_cell_list = []
        for nn_info in range(self._config['layer_num']):
            # rnn_cell = tf.contrib.rnn.BasicGRUCell(self._hidden_size, forget_bias=0.0, state_is_tuple=True)
            rnn_cell = tf.contrib.rnn.GRUCell(self._hidden_size)

            label_rnn_cell_list.append(rnn_cell)

        label_cell = tf.contrib.rnn.MultiRNNCell(label_rnn_cell_list, state_is_tuple=True)


        word_embedding = tf.get_variable("word_embedding", [self._word_vocab_size, self._hidden_size],
                                         dtype=data_type())
        label_embedding = tf.get_variable("label_embedding", [self._label_vocab_size, self._hidden_size],
                                          dtype=data_type())
        word_inputs = tf.nn.embedding_lookup(word_embedding, input_data[:, :2])
        label_inputs = tf.nn.embedding_lookup(label_embedding, input_data[:, 3:])


        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
        inputs = tf.concat(
            [word_inputs[:, 0:1], label_inputs[:, 0:1], word_inputs[:, 1:2], label_inputs[:, 1:2]], axis=1)
        words_targets = input_data[:, 1:3]
        labels_targets = input_data[:, 3:5]
        state = word_cell.zero_state(batch_size, tf.float32)
        outputs = []
        for i in range(self._config["sequence_length"] - 1):
            if i % 2 == 0:
                output, state = word_cell(inputs[:, i], state)
            else:
                output, state = label_cell(inputs[:, i], state)
            outputs.append(output)
            # print(output.shape)
        # print(outputs)
        outputs = tf.transpose(outputs, perm=[1, 0, 2])

        words_outputs = tf.concat([outputs[:, 1:2], outputs[:, 3:4]], axis=1)
        labels_outputs = tf.concat([outputs[:, 0:1], outputs[:, 2:3]], axis=1)
        # print(words_outputs.shape, labels_outputs.shape)

        words_outputs = tf.reshape(words_outputs, [-1, self._hidden_size])
        labels_outputs = tf.reshape(labels_outputs, [-1, self._hidden_size])

        word_softmax_w = tf.get_variable(
            "word_softmax_w", [self._hidden_size, self._word_vocab_size], dtype=data_type())
        word_softmax_b = tf.get_variable("word_softmax_b", [self._word_vocab_size], dtype=data_type())

        label_softmax_w = tf.get_variable(
            "label_softmax_w", [self._hidden_size, self._label_vocab_size], dtype=data_type())
        label_softmax_b = tf.get_variable("label_softmax_b", [self._label_vocab_size], dtype=data_type())

        words_y_flat = tf.reshape(words_targets, [-1])
        labels_y_flat = tf.reshape(labels_targets, [-1])
        words_logits = tf.matmul(words_outputs, word_softmax_w) + word_softmax_b
        labels_logits = tf.matmul(labels_outputs, label_softmax_w) + label_softmax_b
        # print(words_logits, words_targets)
        # print(labels_logits, labels_targets)
        words_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=words_logits, labels=words_y_flat)
        labels_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=labels_logits, labels=labels_y_flat)

        words_losses = tf.reshape(words_losses, [batch_size, -1])
        labels_losses = tf.reshape(labels_losses, [batch_size, -1])
        cost = - tf.concat([words_losses, labels_losses], axis=1)
        # cost = tf.reduce_mean([words_losses, labels_losses])

        return cost

    @property
    def cost_op(self):
        return self._cost_op

    @property
    def data_placeholder(self):
        return self._data_placeholder

