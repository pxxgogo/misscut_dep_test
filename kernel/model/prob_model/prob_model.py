import plyvel
import re
import os
import struct
import json
import redis
import sys
from kernel.model.prob_model.word_embedding import Word_vectors

NUMBER_RE_COMPILOR = re.compile(r"[\.-]?\d[0-9\.%-]*")
NUMBER_TAG = "{{#}}"
NUMBER_TAG_LENGTH = len(NUMBER_TAG)
LETTER_RE_COMPILOR = re.compile(r"[a-zA-Z][a-zA-Z\.'-]*")
LETTER_TAG = "{{E}}"
LETTER_TAG_LENGTH = len(LETTER_TAG)
CONFIG_DIR = "config.json"
FAST_STAT_DB_FLAG = 2
SMOOTH_VECTOR_SCORE_THRESHOLD = 0.6
SMOOTH_FLAG = False
SMOOTH_VALUE_THRESHOLD = 80000


def replace_special_symbols(sentence):
    # modified_sentence = NUMBER_RE_COMPILOR.sub(NUMBER_TAG, sentence)
    # modified_sentence = LETTER_RE_COMPILOR.sub(LETTER_TAG, modified_sentence)
    return sentence


def i_2_b(digit):
    # print(digit, digit.to_bytes(8, byteorder='big'))
    return digit.to_bytes(8, byteorder='big')


def b_2_i(bytes):
    # print(bytes, struct.unpack(">q", bytes))
    return struct.unpack(">q", bytes)[0]


def b_2_i_redis(bytes):
    index = len(bytes) - 1
    seed = 1
    ret = 0
    while index >= 0:
        ret += seed * (bytes[index] - 48)
        seed *= 10
        index -= 1
    return ret


class SmoothCache:
    def __init__(self):
        self._table = {}
        self._size = 0
        self._key_list = []
        self._max_size = 100000
        self._index = 0

    def add(self, key, value):
        if key in self._table:
            return
        if self._size < self._max_size:
            self._table[key] = value
            self._key_list.append(key)
            self._size += 1
        else:
            replaced_key = self._key_list[self._index]
            self._key_list[self._index] = key
            self._table.pop(replaced_key)
            self._table[key] = value
            self._index += 1
            if self._index == self._size:
                self._index = 0

    def get(self, key):
        return self._table.get(key, None)


class ProbModel:
    instance = None

    @classmethod
    def get_instance(cls):
        if cls.instance:
            return cls.instance
        else:
            cls.instance = cls()
            return cls.instance

    def __init__(self, gpu_flag=0):
        config_dir = CONFIG_DIR
        with open(config_dir) as handle:
            config = json.load(handle)
        model_dir = config["stat_models_dir"]
        # deep_model_dir = os.path.join(model_dirs["root_dir"], model_dirs["triples_deep_dir"])
        # broad_model_dir = os.path.join(model_dirs["root_dir"], model_dirs["triples_broad_dir"])
        # single_dep_model_dir = os.path.join(model_dirs["root_dir"], model_dirs["bigrams_dir"])
        # fre_model_dir = os.path.join(model_dirs["root_dir"], model_dirs["unigrams_dir"])

        self._deep_dbs = {'s1': plyvel.DB(os.path.join(model_dir, 'td-s1.db')),
                          's2': plyvel.DB(os.path.join(model_dir, 'td-s2.db')),
                          's3': plyvel.DB(os.path.join(model_dir, 'td-s3.db')),
                          'b12': plyvel.DB(os.path.join(model_dir, 'td-b12.db')),
                          'b13': plyvel.DB(os.path.join(model_dir, 'td-b13.db')),
                          'b23': plyvel.DB(os.path.join(model_dir, 'td-b23.db')),
                          't123': plyvel.DB(os.path.join(model_dir, 'td-t123.db'))}

        self._broad_dbs = {'s1': plyvel.DB(os.path.join(model_dir, 'tb-s1.db')),
                           's2': plyvel.DB(os.path.join(model_dir, 'tb-s2.db')),
                           's3': plyvel.DB(os.path.join(model_dir, 'tb-s3.db')),
                           'b12': plyvel.DB(os.path.join(model_dir, 'tb-b12.db')),
                           'b13': plyvel.DB(os.path.join(model_dir, 'tb-b13.db')),
                           'b23': plyvel.DB(os.path.join(model_dir, 'tb-b23.db')),
                           't123': plyvel.DB(os.path.join(model_dir, 'tb-t123.db'))}
        self._single_dbs = {'s1': plyvel.DB(os.path.join(model_dir, 'b-s1.db')),
                            's2': plyvel.DB(os.path.join(model_dir, 'b-s2.db')),
                            'b12': plyvel.DB(os.path.join(model_dir, 'b-b12.db'))}
        self._fre_db = plyvel.DB(os.path.join(model_dir, 'fre.db'))

        if FAST_STAT_DB_FLAG == 1:
            self._redis_db = redis.StrictRedis(host='localhost', port=6379, db=0)
        elif FAST_STAT_DB_FLAG == 2:
            sys.path.append(config["cpp_stat_model_dir"])
            import prob_model
            self._prob_cpp_db = prob_model.Prob_model_CPP()
            self._prob_cpp_db.load_model(os.path.join(config["cpp_stat_model_dir"], config["cpp_stat_model_name"]))
        self._model_types = {0: "s1", 1: "s2", 2: "s3", 3: "b12", 4: "b13", 5: "b23", 6: "t123"}
        self._single_dep_model_types = {0: "s-s1", 1: "s-s2", 2: "s-b12"}

        if SMOOTH_FLAG:
            word_embedding_dir = config["word_embedding_dir"]
            self._word_embedding = Word_vectors(word_embedding_dir, gpu_flag=gpu_flag)
            self._smooth_cache = SmoothCache()

        # self._fast_db_timer = 0
        # self._all_db_timer = 0
        # self._fast_db_times = 0
        # self._all_db_times = 0

    def _get_value(self, key, db_type):
        fast_db_key = "%s %s" % (db_type, key)
        # start_time = time.time()
        if FAST_STAT_DB_FLAG == 1:
            value_fast_db = self._redis_db.get(fast_db_key)
        elif FAST_STAT_DB_FLAG == 2:
            value_fast_db = self._prob_cpp_db.get(fast_db_key)
        else:
            value_fast_db = None
        # end_time = time.time()
        #
        # self._fast_db_timer += end_time - start_time
        # self._all_db_timer += end_time - start_time
        # self._fast_db_times += 1
        # self._all_db_times += 1

        if value_fast_db is not None:
            if FAST_STAT_DB_FLAG == 1:
                return b_2_i_redis(value_fast_db)
            elif FAST_STAT_DB_FLAG == 2:
                if value_fast_db != 0:
                    return value_fast_db
        key = key.encode("utf-8")
        # start_time = time.time()
        if db_type == "fre":
            value = self._fre_db.get(key)
        else:
            db_sub_type = db_type[2:]
            db_main_type = db_type[0]
            if db_main_type == 's':
                value = self._single_dbs[db_sub_type].get(key)
            elif db_main_type == 'd':
                value = self._deep_dbs[db_sub_type].get(key)
            elif db_main_type == 'b':
                value = self._broad_dbs[db_sub_type].get(key)
            else:
                value = None
        # end_time = time.time()
        #
        # self._all_db_timer += end_time - start_time
        # self._all_db_times += 1

        if value is None:
            score = 0
        else:
            score = b_2_i(value)
        return score

    def _search_smooth_value(self, key, rets):
        self._prob_cpp_db.search(key, rets)

    def get_smooth_score(self, db_main_type_No, dep_key, model_type_No, modified_words):
        if db_main_type_No == 0:
            main_key = "d-t123"
        else:
            main_key = "b-t123"
        if model_type_No == 2:
            key = "%s %s: %s %s .*" % (main_key, dep_key, modified_words[0], modified_words[1])
            main_word = modified_words[2]
        elif model_type_No == 1:
            key = "%s %s: %s .* %s" % (main_key, dep_key, modified_words[0], modified_words[2])
            main_word = modified_words[1]
        elif model_type_No == 0:
            key = "%s %s: .* %s %s" % (main_key, dep_key, modified_words[1], modified_words[2])
            main_word = modified_words[0]
        else:
            key = ""
            main_vector = -1
            return 0
        if not self._word_embedding.is_in_vocab(main_word):
            return 0
        rets = []
        self._search_smooth_value(key, rets)
        ret_score = 0
        words = []
        values = []
        smooth_keys = []
        for ret in rets:
            word = ret[0]
            smooth_key = "%s %s" % (word, main_word)
            similar_value = self._smooth_cache.get(smooth_key)
            if similar_value:
                if similar_value > SMOOTH_VECTOR_SCORE_THRESHOLD:
                    ret_score += ret[1]
                    continue
            else:
                if not self._word_embedding.is_in_vocab(word):
                    continue
                words.append(word)
                values.append(ret[1])
                smooth_keys.append(smooth_key)
        if len(words) == 0:
            return ret_score
        similarities = self._word_embedding.get_similarities(main_word, words)
        for value, similarity, smooth_key in zip(values, similarities, smooth_keys):
            self._smooth_cache.add(smooth_key, similarity)
            if similarity > SMOOTH_VECTOR_SCORE_THRESHOLD:
                ret_score += value
        del words
        del values
        del rets
        del smooth_keys
        del similarities
        return ret_score

    def _score(self, db_main_type_No, word_1_info, label_1, word_2_info, label_2, word_3_info):
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

        smooth_model_type_flags = []

        scores = []
        score = self._get_value(modified_word_1, "fre")
        if score < SMOOTH_VALUE_THRESHOLD:
            smooth_model_type_flags.append((0, True))
        else:
            smooth_model_type_flags.append((0, False))
        scores.append(score)

        score = self._get_value(modified_word_2, "fre")
        if score < SMOOTH_VALUE_THRESHOLD:
            smooth_model_type_flags.append((1, True))
        else:
            smooth_model_type_flags.append((1, False))
        scores.append(score)

        score = self._get_value(modified_word_3, "fre")
        if score < SMOOTH_VALUE_THRESHOLD:
            smooth_model_type_flags.append((2, True))
        else:
            smooth_model_type_flags.append((2, False))
        scores.append(score)

        if db_main_type_No == 0:
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
                db_type = self._single_dep_model_types[model_type_No % 3]
                score = self._get_value(key, db_type)
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
                db_type = self._single_dep_model_types[model_type_No % 3]
                score = self._get_value(key, db_type)
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
            db_sub_type = self._model_types[model_type_No]
            if db_main_type_No == 0:
                db_type = "d-" + db_sub_type
            else:
                db_type = "b-" + db_sub_type
            score = self._get_value(key, db_type)
            scores.append(score)

        if not SMOOTH_FLAG:
            return scores

        for i, smooth_feature_flag in smooth_model_type_flags:
            if smooth_feature_flag and SMOOTH_FLAG:
                score = self.get_smooth_score(db_main_type_No, dep_key, i,
                                              (modified_word_1, modified_word_2, modified_word_3))
            else:
                score = 0
            scores.append(score)
        return scores

    # def init_timer(self):
    #     self._fast_db_timer = self._all_db_timer = 0
    #     self._fast_db_times = self._all_db_times = 0
    #
    #
    # def get_timer(self):
    #     return self._all_db_timer, self._fast_db_timer, self._all_db_times, self._fast_db_times

    def get_data(self, data):
        rets = []
        for sub_data in data:
            ret = self._score(*sub_data)
            rets.append(ret)
        return rets

    def get_fre_value(self, word):
        modified_word = replace_special_symbols(word)
        score = self._get_value(modified_word, 'fre')
        return score
