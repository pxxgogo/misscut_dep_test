import plyvel
import re
import os
import struct
from .word_embedding import Word_vectors

NUMBER_RE_COMPILOR = re.compile(r"[\.-]?\d[0-9\.%-]*")
NUMBER_TAG = "{{#}}"
NUMBER_TAG_LENGTH = len(NUMBER_TAG)
LETTER_RE_COMPILOR = re.compile(r"[a-zA-Z][a-zA-Z\.'-]*")
LETTER_TAG = "{{E}}"
LETTER_TAG_LENGTH = len(LETTER_TAG)


def replace_special_symbols(sentence):
    modified_sentence = NUMBER_RE_COMPILOR.sub(NUMBER_TAG, sentence)
    modified_sentence = LETTER_RE_COMPILOR.sub(LETTER_TAG, modified_sentence)
    return modified_sentence


def i_2_b(digit):
    # print(digit, digit.to_bytes(8, byteorder='big'))
    return digit.to_bytes(8, byteorder='big')


def b_2_i(bytes):
    # print(bytes, struct.unpack(">q", bytes))
    return struct.unpack(">q", bytes)[0]


class ProbModel:
    def __init__(self, deep_model_dir="./model/leveldb/deep", broad_model_dir="./model/leveldb/broad",
                 fre_model_dir="./model/leveldb/word_frequency/word_frequency.db",
                 single_dep_model_dir="./model/leveldb/single_dep", embedding_dir="./model/embedding/cc.zh.300.vec"):
        self._deep_dbs = {'s1': plyvel.DB(os.path.join(deep_model_dir, 's1.db')),
                          's2': plyvel.DB(os.path.join(deep_model_dir, 's2.db')),
                          's3': plyvel.DB(os.path.join(deep_model_dir, 's3.db')),
                          'b12': plyvel.DB(os.path.join(deep_model_dir, 'b12.db')),
                          'b13': plyvel.DB(os.path.join(deep_model_dir, 'b13.db')),
                          'b23': plyvel.DB(os.path.join(deep_model_dir, 'b23.db')),
                          't123': plyvel.DB(os.path.join(deep_model_dir, 't123.db'))}

        self._broad_dbs = {'s1': plyvel.DB(os.path.join(broad_model_dir, 's1.db')),
                           's2': plyvel.DB(os.path.join(broad_model_dir, 's2.db')),
                           's3': plyvel.DB(os.path.join(broad_model_dir, 's3.db')),
                           'b12': plyvel.DB(os.path.join(broad_model_dir, 'b12.db')),
                           'b13': plyvel.DB(os.path.join(broad_model_dir, 'b13.db')),
                           'b23': plyvel.DB(os.path.join(broad_model_dir, 'b23.db')),
                           't123': plyvel.DB(os.path.join(broad_model_dir, 't123.db'))}
        self._single_dbs = {'s1': plyvel.DB(os.path.join(single_dep_model_dir, 's1.db')),
                            's2': plyvel.DB(os.path.join(single_dep_model_dir, 's2.db')),
                            'b12': plyvel.DB(os.path.join(single_dep_model_dir, 'b12.db'))}
        self._fre_db = plyvel.DB(fre_model_dir)

        self._model_types = {0: "s1", 1: "s2", 2: "s3", 3: "b12", 4: "b13", 5: "b23", 6: "t123"}
        self._single_dep_model_types = {0: "s1", 1: "s2", 2: "b12"}
        self.word_embedding = Word_vectors(embedding_dir, 200000)

    def get_data(self, data):
        rets = []
        for sub_data in data:
            ret = self.score(*sub_data)
            rets.append(ret)
        return rets

    def score(self, type, word_1_info, label_1, word_2_info, label_2, word_3_info):
        if type == 0:
            dbs = self._deep_dbs
        else:
            dbs = self._broad_dbs
        dep_key = "%s %s" % (label_1, label_2)
        word_1 = word_1_info["text"]
        word_2 = word_2_info["text"]
        word_3 = word_3_info["text"]
        if word_1 != "{ROOT}":
            modified_word_1 = replace_special_symbols(word_1)
        else:
            modified_word_1 = word_1
        modified_word_2 = replace_special_symbols(word_2)
        modified_word_3 = replace_special_symbols(word_3)

        scores = []
        key = modified_word_1.encode("utf-8")
        value = self._fre_db.get(key)
        if value is None:
            score = 0
        else:
            score = b_2_i(value)
        word_fre_1 = score
        scores.append(score)

        key = modified_word_2.encode("utf-8")
        value = self._fre_db.get(key)
        if value is None:
            score = 0
        else:
            score = b_2_i(value)
        word_fre_2 = score
        scores.append(score)

        key = modified_word_3.encode("utf-8")
        value = self._fre_db.get(key)
        if value is None:
            score = 0
        else:
            score = b_2_i(value)
        word_fre_3 = score
        scores.append(score)

        if type == 0:
            for model_type_No in range(6):
                if model_type_No == 0:
                    key = "%s: %s" % (label_1, modified_word_1)
                elif model_type_No == 1:
                    key = "%s: %s" % (label_1, modified_word_2)
                elif model_type_No == 2:
                    key = "%s: %s %s" % (label_1, modified_word_1, modified_word_2)
                elif model_type_No == 3:
                    key = "%s: %s" % (label_2, modified_word_2)
                elif model_type_No == 4:
                    key = "%s: %s" % (label_2, modified_word_3)
                elif model_type_No == 5:
                    key = "%s: %s %s" % (label_2, modified_word_2, modified_word_3)
                else:
                    key = ""
                db = self._single_dbs[self._single_dep_model_types[model_type_No % 3]]
                key = key.encode("utf-8")
                value = db.get(key)
                if value is None:
                    score = 0
                else:
                    score = b_2_i(value)
                scores.append(score)
        else:
            for model_type_No in range(6):
                if model_type_No == 0:
                    key = "%s: %s" % (label_1, modified_word_1)
                elif model_type_No == 1:
                    key = "%s: %s" % (label_1, modified_word_2)
                elif model_type_No == 2:
                    key = "%s: %s %s" % (label_1, modified_word_1, modified_word_2)
                elif model_type_No == 3:
                    key = "%s: %s" % (label_2, modified_word_1)
                elif model_type_No == 4:
                    key = "%s: %s" % (label_2, modified_word_3)
                elif model_type_No == 5:
                    key = "%s: %s %s" % (label_2, modified_word_1, modified_word_3)
                else:
                    key = ""
                db = self._single_dbs[self._single_dep_model_types[model_type_No % 3]]
                key = key.encode("utf-8")
                value = db.get(key)
                if value is None:
                    score = 0
                else:
                    score = b_2_i(value)
                scores.append(score)


        for model_type_No in range(7):
            if model_type_No == 0:
                key = "%s: %s" % (dep_key, modified_word_1)
            elif model_type_No == 1:
                key = "%s: %s" % (dep_key, modified_word_2)
            elif model_type_No == 2:
                key = "%s: %s" % (dep_key, modified_word_3)
            elif model_type_No == 3:
                key = "%s: %s %s" % (dep_key, modified_word_1, modified_word_2)
            elif model_type_No == 4:
                key = "%s: %s %s" % (dep_key, modified_word_1, modified_word_3)
            elif model_type_No == 5:
                key = "%s: %s %s" % (dep_key, modified_word_2, modified_word_3)
            elif model_type_No == 6:
                key = "%s: %s %s %s" % (dep_key, modified_word_1, modified_word_2, modified_word_3)
            else:
                key = ""
            db = dbs[self._model_types[model_type_No]]
            key = key.encode("utf-8")
            value = db.get(key)
            if value is None:
                score = 0
            else:
                score = b_2_i(value)
            scores.append(score)
        db = dbs['t123']
        similar_score = 0
        if word_fre_1 <= word_fre_2 <= word_fre_3:
            if scores[14] > 0:
                similar_words_info = self.word_embedding.get_closed_words(word_1)
                for similar_word_info in similar_words_info:
                    modified_word = replace_special_symbols(similar_word_info[0])
                    key = "%s: %s %s %s" % (dep_key, modified_word, modified_word_2, modified_word_3)
                    key = key.encode("utf-8")
                    value = db.get(key)
                    if value is None:
                        score = 0
                    else:
                        score = b_2_i(value)
                    similar_score += score
        elif word_fre_2 <= word_fre_1 <= word_fre_3:
            if scores[13] > 0:
                similar_words_info = self.word_embedding.get_closed_words(word_2)
                for similar_word_info in similar_words_info:
                    modified_word = replace_special_symbols(similar_word_info[0])
                    key = "%s: %s %s %s" % (dep_key, modified_word_1, modified_word, modified_word_3)
                    key = key.encode("utf-8")
                    value = db.get(key)
                    if value is None:
                        score = 0
                    else:
                        score = b_2_i(value)
                    similar_score += score
        else:
            if scores[12] > 0:
                similar_words_info = self.word_embedding.get_closed_words(word_3)
                for similar_word_info in similar_words_info:
                    modified_word = replace_special_symbols(similar_word_info[0])
                    key = "%s: %s %s %s" % (dep_key, modified_word_1, modified_word_2, modified_word)
                    key = key.encode("utf-8")
                    value = db.get(key)
                    if value is None:
                        score = 0
                    else:
                        score = b_2_i(value)
                    similar_score += score
        scores.append(similar_score)
        return scores
