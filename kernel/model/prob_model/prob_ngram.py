import plyvel
import re
import os
import struct

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


class ProbNgramModel:
    def __init__(self, model_dir="./model/leveldb/leveldb_ngram", model_flag=0):
        self._dbs = [plyvel.DB(os.path.join(model_dir, '1-gram.db')),
                     plyvel.DB(os.path.join(model_dir, '2-gram.db')),
                     plyvel.DB(os.path.join(model_dir, '3-gram.db'))]

    def score(self, items):
        items.insert(0, "{BOS}")
        items.append("{EOS}")
        item_No = -1
        items_num = len(items)
        rets = []
        while item_No < items_num - 1:
            item_No += 1
            sub_scores = []
            for gram_num in range(1, 4):
                if item_No < gram_num - 1:
                    sub_scores.append(-1)
                    continue
                raw_key = " ".join(items[item_No - gram_num + 1:item_No + 1])
                key = raw_key.encode("utf-8")
                value = self._dbs[gram_num - 1].get(key)
                if value is None:
                    sub_scores.append(0)
                else:
                    sub_scores.append(b_2_i(value))
            rets.append((items[item_No], sub_scores))
        return rets
