import kenlm
import re


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

class Ngram:
    def __init__(self, model_flag=0):
        self.model_flag = model_flag
        if model_flag == 0:
            self._broad_model = kenlm.LanguageModel("./model/gigaword_broad.bin")
            self._deep_model = kenlm.LanguageModel("./model/gigaword_deep.bin")
        else:
            self._deep_model = kenlm.LanguageModel("./model/ret_tuple_deep.bin")
            self._broad_model = kenlm.LanguageModel("./model/ret_tuple_broad.bin")

    def score(self, word_0, label_0, word_1, label_1, word_2, type):
        if word_0 != "{ROOT}":
            modified_word_0 = replace_special_symbols(word_0)
        else:
            modified_word_0 = word_0
        modified_word_1 = replace_special_symbols(word_1)
        modified_word_2 = replace_special_symbols(word_2)
        if self.model_flag == 0:
            sentence = "%s %s %s %s %s" % (modified_word_0, label_0, modified_word_1, label_1, modified_word_2)
            score_num = 6
        else:
            sentence = "%s %s %s" % (modified_word_0, modified_word_1, modified_word_2)
            score_num = 4
        # print(sentence)
        if type == 0:
            return list(self._deep_model.full_scores(sentence))
            # return [(0.0, 0) for i in range(score_num)]
        elif type == 1:
            return list(self._broad_model.full_scores(sentence))
        else:
            return []



