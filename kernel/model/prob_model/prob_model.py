import plyvel
import re
import os
import struct
from .word_vectors import Word_vectors

NUMBER_RE_COMPILOR = re.compile(r"[\.-]?\d[0-9\.%-]*")
NUMBER_TAG = "{{#}}"
NUMBER_TAG_LENGTH = len(NUMBER_TAG)
LETTER_RE_COMPILOR = re.compile(r"[a-zA-Z][a-zA-Z\.'-]*")
LETTER_TAG = "{{E}}"
LETTER_TAG_LENGTH = len(LETTER_TAG)
ANALYZED_THRESHOLD = 100


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
                 word_vectors_dir="./model/leveldb/word_vectors.txt", model_flag=0):
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
        self._model_types = {0: "s1", 1: "s2", 2: "s3", 3: "b12", 4: "b13", 5: "b23", 6: "t123"}

        self.word_vectors = Word_vectors(word_vectors_dir)

    def get_data(self, data):
        rets = []
        for sub_data in data:
            ret = self.score(*sub_data)
            rets.append(ret)
        return rets

    def get_score(self, model_type_info, dep_key, modified_word_1, modified_word_2, modified_word_3):
        main_model_type = model_type_info[0]
        model_type_No = model_type_info[1]
        if main_model_type == 0:
            dbs = self._deep_dbs
        else:
            dbs = self._broad_dbs
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
        return score

    def generate_ret(self, model_type_No, score, modified_word_1, modified_word_2, modified_word_3):
        if model_type_No == 0:
            ret = (score, (modified_word_1, "X", "X"), model_type_No)
        elif model_type_No == 1:
            ret = (score, ("X", modified_word_2, "X"), model_type_No)
        elif model_type_No == 2:
            ret = (score, ("X", "X", modified_word_3), model_type_No)
        elif model_type_No == 3:
            ret = (score, (modified_word_1, modified_word_2, "X"), model_type_No)
        elif model_type_No == 4:
            ret = (score, (modified_word_1, "X", modified_word_3), model_type_No)
        elif model_type_No == 5:
            ret = (score, ("X", modified_word_2, modified_word_3), model_type_No)
        elif model_type_No == 6:
            ret = (score, (modified_word_1, modified_word_2, modified_word_3), model_type_No)
        else:
            ret = ()
        return ret

    def score(self, type, word_1, label_1, word_2, label_2, word_3):

        dep_key = "%s %s" % (label_1, label_2)
        if word_1 != "{ROOT}":
            modified_word_1 = replace_special_symbols(word_1)
        else:
            modified_word_1 = word_1
        modified_word_2 = replace_special_symbols(word_2)
        modified_word_3 = replace_special_symbols(word_3)
        rets = []
        for model_type_No in range(7):
            score = self.get_score((type, model_type_No), dep_key, modified_word_1, modified_word_2, modified_word_3)
            rets.append(self.generate_ret(model_type_No, score, modified_word_1, modified_word_2, modified_word_3))
        self.analyze_rets(rets, (modified_word_1, modified_word_2, modified_word_3), dep_key, type)
        rets.sort(key=lambda x: x[2])
        return rets

    def analyze_rets(self, rets, words, dep_key, main_model_type):
        '''
            规则：
            当一元出现一个稀疏时，找到其接近的词集，之后用词集中的词代替该词进行后续的统计
        '''
        analyze_flag = False
        analyzed_word_No = -1
        for i in range(3):
            if rets[i][0] < ANALYZED_THRESHOLD:
                if analyzed_word_No == -1:
                    analyze_flag = True
                    analyzed_word_No = i
                else:
                    analyze_flag = False
                    analyzed_word_No = -1
        if not analyze_flag:
            return
        word = words[analyzed_word_No]
        closed_words = self.word_vectors.get_closed_words(word)
        if len(closed_words) == 0:
            return
        for model_type_No in range(4):
            for word in closed_words:
                modified_word = replace_special_symbols(word[0])
                if analyzed_word_No == 0:
                    score = self.get_score((main_model_type, model_type_No + 3), dep_key, modified_word, words[1],
                                           words[2])
                    if score > 0:
                        rets.append(self.generate_ret(model_type_No, score, modified_word, words[1], words[2]))
                elif analyzed_word_No == 1:
                    score = self.get_score((main_model_type, model_type_No + 3), dep_key, words[0], modified_word,
                                           words[2])
                    if score > 0:
                        rets.append(self.generate_ret(model_type_No, score, words[0], modified_word, words[2]))
                elif analyzed_word_No == 2:
                    score = self.get_score((main_model_type, model_type_No + 3), dep_key, words[0], words[1],
                                           modified_word)
                    if score > 0:
                        rets.append(self.generate_ret(model_type_No, score, words[0], words[1], modified_word))
