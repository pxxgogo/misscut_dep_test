import json
import tensorflow as tf
from .rnn import RNN

WORDS_VOCAB = "words_vocab"
LABELS_VOCAB = "labels_vocab"


class RNN_wrapper:

    def __init__(self):
        with open("./kernel/model/rnn/config.json") as handle:
            self._config = json.load(handle)
        with open("./kernel/model/rnn/ret.vocab") as handle:
            self._vocabs = json.load(handle)
        self._words_OOV = len(self._vocabs[WORDS_VOCAB])
        self._labels_OOV = len(self._vocabs[LABELS_VOCAB])
        session_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        initializer = tf.random_uniform_initializer(-0.4, 0.4)

        self._deep_session = tf.Session(config=session_config)
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE, initializer=initializer):
            self._deep_model = RNN(config=self._config)
        self._deep_session.run(tf.global_variables_initializer())
        new_saver = tf.train.Saver()
        new_saver.restore(self._deep_session, tf.train.latest_checkpoint(
            self._config["deep_model_dir"]))

        self._broad_session = tf.Session(config=session_config)
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE, initializer=initializer):
            self._broad_model = RNN(config=self._config)
        self._broad_session.run(tf.global_variables_initializer())
        new_saver = tf.train.Saver()
        new_saver.restore(self._broad_session, tf.train.latest_checkpoint(
            self._config["broad_model_dir"]))

    def get_words_labels_ID(self, data):
        ids = [self._vocabs[WORDS_VOCAB].get(data[1], self._words_OOV),
               self._vocabs[WORDS_VOCAB].get(data[3], self._words_OOV),
               self._vocabs[WORDS_VOCAB].get(data[5], self._words_OOV),
               self._vocabs[LABELS_VOCAB].get(data[2], self._labels_OOV),
               self._vocabs[LABELS_VOCAB].get(data[4], self._labels_OOV)]
        return ids

    def rearrange_rets(self, deep_costs, broad_costs, deep_data_buffer, broad_data_buffer, data):
        rets = []
        deep_No = 0
        broad_No = 0
        for sub_data in data:
            item_type = sub_data[0]
            if item_type == 0:
                costs = deep_costs[deep_No]
                data = deep_data_buffer[deep_No][1:]
                ret = []
                for i, ID, cost in zip(range(4), data, costs):
                    if i < 2:
                        if ID == self._words_OOV:
                            ret.append((cost, True))
                        else:
                            ret.append((cost, False))
                    else:
                        if ID == self._labels_OOV:
                            ret.append((cost, True))
                        else:
                            ret.append((cost, False))
                rets.append([ret[2], ret[0], ret[3], ret[1]])
                deep_No += 1
            else:
                costs = broad_costs[broad_No]
                data = broad_data_buffer[broad_No][1:]
                ret = []
                for i, ID, cost in zip(range(4), data, costs):
                    if i < 2:
                        if ID == self._words_OOV:
                            ret.append((cost, True))
                        else:
                            ret.append((cost, False))
                    else:
                        if ID == self._labels_OOV:
                            ret.append((cost, True))
                        else:
                            ret.append((cost, False))
                rets.append([ret[2], ret[0], ret[3], ret[1]])

                broad_No += 1
        return rets

    def get_data(self, data):
        deep_data_buffer = []
        broad_data_buffer = []
        for sub_data in data:
            item_type = sub_data[0]
            IDs = self.get_words_labels_ID(sub_data)
            if item_type == 0:
                deep_data_buffer.append(IDs)
            else:
                broad_data_buffer.append(IDs)
        if len(deep_data_buffer) > 0:
            deep_costs = self._deep_session.run(self._deep_model.cost_op,
                                                feed_dict={self._deep_model.data_placeholder: deep_data_buffer})
        else:
            deep_costs = []
        if len(broad_data_buffer) > 0:
            broad_costs = self._broad_session.run(self._broad_model.cost_op,
                                                  feed_dict={self._broad_model.data_placeholder: broad_data_buffer})
        else:
            broad_costs = []
        # print(broad_costs, deep_costs)
        scores = self.rearrange_rets(deep_costs, broad_costs, deep_data_buffer, broad_data_buffer, data)
        return scores
