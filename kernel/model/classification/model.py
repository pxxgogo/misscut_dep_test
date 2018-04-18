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
    return tf.float32


class Model(object):
    def __init__(self, config):
        self._batch_size = config['batch_size']
        self._hidden_size = config['hidden_size']
        self._word_vocab_size = config['word_vocab_size']
        self._label_vocab_size = config['label_vocab_size']
        self._config = config
        self._data_placeholder = tf.placeholder(tf.int32, [None, self._config["sequence_length"]])
        self._logits = self.calculate_cost()

    def calculate_cost(self):
        input_data = self._data_placeholder
        sub_batch_size = self._batch_size // self._config["gpu_num"]
        word_embedding = tf.get_variable("word_embedding", [self._word_vocab_size, self._hidden_size],
                                         dtype=data_type())
        label_embedding = tf.get_variable("label_embedding", [self._label_vocab_size, self._hidden_size],
                                          dtype=data_type())
        word_inputs = tf.nn.embedding_lookup(word_embedding, input_data[:, :3])
        label_inputs = tf.nn.embedding_lookup(label_embedding, input_data[:, 3:5])

        data = tf.concat(
            [word_inputs[:, 0:1], label_inputs[:, 0:1], word_inputs[:, 1:2], label_inputs[:, 1:2], word_inputs[:, 2:3]],
            axis=1)
        nn_infos = self._config['nn_infos']
        layer_No = 0
        for nn_info in nn_infos:
            if nn_info["net_type"] == "CONV":
                if len(data.shape) == 3:
                    data = tf.reshape(data, [data.shape[0], data.shape[1], data.shape[2], 1])
                for i in range(nn_info["repeated_times"]):
                    data = self.add_conv_layer(layer_No, data, nn_info["filter_size"], nn_info["out_channels"],
                                               nn_info["filter_type"], self._config["regularized_lambda"],
                                               self._config["regularized_flag"])
                    layer_No += 1
            elif nn_info["net_type"] == "POOL":
                if len(data.shape) == 3:
                    data = tf.reshape(data, [data.shape[0], data.shape[1], data.shape[2], 1])
                for i in range(nn_info["repeated_times"]):
                    data = self.add_pool_layer(layer_No, data, nn_info["pool_size"], nn_info["pool_type"])
                    layer_No += 1
            elif nn_info["net_type"] == "DENSE":
                for i in range(nn_info["repeated_times"]):
                    data = self.add_dense_layer(layer_No, data, nn_info["output_size"], nn_info["keep_prob"],
                                                self._config["regularized_lambda"], self._config["regularized_flag"])
                    layer_No += 1

        data = tf.reshape(data, [sub_batch_size, -1])
        softmax_w = tf.get_variable(
            "softmax_w", [data.shape[1], 2], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [2], dtype=data_type())
        logits = tf.matmul(data, softmax_w) + softmax_b
        return logits

    def add_conv_layer(self, No, input, filter_size, out_channels, filter_type, regularized_lambda, r_flag=True,
                       strides=[1, 1, 1, 1]):
        with tf.variable_scope("conv_layer_%d" % No):
            W = tf.get_variable('filter', [filter_size[0], filter_size[1], input.shape[3], out_channels])
            if r_flag:
                tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularized_lambda)(W))
            b = tf.get_variable('bias', [out_channels])
            conv = tf.nn.conv2d(
                input,
                W,
                strides=strides,
                padding=filter_type,
                name='conv'
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        return h

    def add_pool_layer(self, No, input, pool_size, pool_type, strides=[1, 1, 1, 1]):
        for i in range(2):
            if pool_size[i] == -1:
                pool_size[i] = input.shape[1 + i]
        with tf.variable_scope("pool_layer_%d" % No):
            pooled = tf.nn.max_pool(
                input,
                ksize=[1, pool_size[0], pool_size[1], 1],
                padding=pool_type,
                strides=strides,
                name='pool'
            )
        return pooled

    def get_length(self, input):
        ret = 1
        for i in range(1, len(input.shape)):
            ret *= int(input.shape[i])
        return ret

    def add_dense_layer(self, No, input, output_size, keep_prob, regularized_lambda, r_flag=True):
        with tf.variable_scope("dense_layer_%d" % No):
            input_length = self.get_length(input)
            W = tf.get_variable('dense', [input_length, output_size])
            if r_flag:
                tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularized_lambda)(W))
            b = tf.get_variable('bias', [output_size])
            data = tf.reshape(input, [-1, int(input_length)])
            data = tf.nn.relu(tf.matmul(data, W) + b)
            if keep_prob < 1.0:
                data = tf.nn.dropout(data, keep_prob)
        return data

    @property
    def dataset_iterator(self):
        return self._dataset_iterator

    @property
    def logits(self):
        return self._logits

    @property
    def data_placeholder(self):
        return self._data_placeholder
