import json
import tensorflow as tf
from .model import Model

WORDS_VOCAB = "words_vocab"
LABELS_VOCAB = "labels_vocab"


class Wrapper:

    def __init__(self):
        with open("./kernel/model/classification/config_CG.json") as handle:
            self._config = json.load(handle)
        with open("./kernel/model/classification/ret_CG.vocab") as handle:
            self._vocabs = json.load(handle)
        self._words_OOV = len(self._vocabs[WORDS_VOCAB])
        self._labels_OOV = len(self._vocabs[LABELS_VOCAB])
        session_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        initializer = tf.random_uniform_initializer(-0.4, 0.4)

        self._deep_session = tf.Session(config=session_config)
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE, initializer=initializer):
            self._deep_model = Model(config=self._config)
        self._deep_session.run(tf.global_variables_initializer())
        new_saver = tf.train.Saver()
        new_saver.restore(self._deep_session, tf.train.latest_checkpoint(
            self._config["deep_model_dir"]))

        self._broad_session = tf.Session(config=session_config)
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE, initializer=initializer):
            self._broad_model = Model(config=self._config)
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

    def rearrange_rets(self, deep_rets, broad_rets, deep_data_buffer, broad_data_buffer, data):
        rets = []
        deep_No = 0
        broad_No = 0
        # print(len(deep_rets))
        for sub_data in data:
            item_type = sub_data[0]
            if item_type == 0:
                # print(deep_rets[deep_No])
                data = deep_data_buffer[deep_No]
                OOV_flag = False
                for i, ID, in zip(range(4), data):
                    if i < 2:
                        if ID == self._words_OOV:
                            OOV_flag = True
                    else:
                        if ID == self._labels_OOV:
                            OOV_flag = True
                rets.append((deep_rets[deep_No], OOV_flag))
                deep_No += 1
            else:
                # print(broad_rets[broad_No])
                data = broad_data_buffer[broad_No]
                OOV_flag = False
                for i, ID, in zip(range(4), data):
                    if i < 2:
                        if ID == self._words_OOV:
                            OOV_flag = True
                    else:
                        if ID == self._labels_OOV:
                            OOV_flag = True
                rets.append((broad_rets[broad_No], OOV_flag))
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
            deep_rets= self._deep_session.run(self._deep_model.logits,
                                          feed_dict={self._deep_model.data_placeholder: deep_data_buffer})
            # print("D", deep_rets, deep_data_buffer)
        else:
            deep_rets = []
        if len(broad_data_buffer) > 0:
            broad_rets = self._broad_session.run(self._broad_model.logits,
                                          feed_dict={self._broad_model.data_placeholder: broad_data_buffer})
            # print("B", broad_rets, broad_data_buffer)
        else:
            broad_rets = []
        # print(broad_rets, deep_rets)
        rets = self.rearrange_rets(deep_rets, broad_rets, deep_data_buffer, broad_data_buffer, data)
        return rets
