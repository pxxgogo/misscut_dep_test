import re
from .model.ngram import ngram
from .model.prob_model import prob_model, prob_ngram
from .model.classification.wrapper import Wrapper
from .model.rnn.wrapper import RNN_wrapper
import kenlm

NGRAM_TYPE = 'ngram'
RNN_TYPE = 'rnn'
PROB_MODEL_TYPE = 'prob_model'
CLASSIFICATION_TYPE = "classification"
PROB_MODEL_NGRAM_TYPE = 'prob_model_ngram'
RECORD_NUM = 50
THRESHOLDS = (-14, -6)

DATA_ID_TUPLE_COMPILER = re.compile("\((\d+), (\d+)\)")


def generate_log_sentence(data, data_type):
    # correct
    if data_type == 0:
        log_sentence = "\n\n\n#CORRECT SENTENCE: %s \n" % (data["correct_sentence"])
    else:
        if data["mistake_type"] in ["7-1", "7-2"]:
            log_sentence = "\n\n\n#WRONG SENTENCE: %s \n" % (
                    data["wrong_sentence"])
        else:
            indexes = data["word_position"]
            log_sentence = "\n\n\n#WRONG SENTENCE: %s \n" % (
                    data["wrong_sentence"][:indexes[0]] + "  @" + data["wrong_word"] + "@  " + data["wrong_sentence"][
                                                                                               indexes[1]:])
    return log_sentence


class Data_container:
    def __init__(self, buffer_size=1, model_type=NGRAM_TYPE, log_path="./log.txt", test_mode=0, gpu_flag=0):
        self._data_buffer = {'data_ID_tuple': [], 'data': []}
        self._buffer_length = 0
        self._log_handle = open(log_path, 'w')
        self._buffer_size = buffer_size
        self._model_type = model_type
        self._main_data_dict = {}
        self._main_token_rets = {}
        self._test_mode = test_mode
        self._data_ID_tuple = -1
        # self.init_debug_ngram_model()
        self.model = None
        self.model_2 = None
        if model_type == NGRAM_TYPE:
            self.model = ngram.Ngram()
        elif model_type == RNN_TYPE:
            self.model = RNN_wrapper()
        elif model_type == PROB_MODEL_TYPE:
            self.model = prob_model.ProbModel(gpu_flag=gpu_flag)
        elif model_type == PROB_MODEL_NGRAM_TYPE:
            self.model = prob_model.ProbModel()
            self.model_2 = prob_ngram.ProbNgramModel()

    def init_buffer(self):
        self._data_buffer = {'data_ID_tuple': [], 'data': []}
        self._buffer_length = 0

    def init_debug_ngram_model(self):
        self._debug_model = kenlm.LanguageModel("./model/gigaword_broad.bin")

    def close_all_handles(self):
        self._log_handle.close()

    def feed_data(self, data, data_ID_tuple, end_flag=False):
        self._data_buffer['data_ID_tuple'].append(data_ID_tuple)
        self._data_buffer['data'].append(data)
        self._buffer_length += 1
        if self._buffer_length == self._buffer_size or end_flag:
            rets = self.model.get_data(self._data_buffer['data'])
            if self._model_type == NGRAM_TYPE:
                self.ngram_ret_operation(rets)
            elif self._model_type == RNN_TYPE:
                self.nn_ret_operation(rets)
            elif self._model_type == PROB_MODEL_TYPE or self._model_type == PROB_MODEL_NGRAM_TYPE:
                self.prob_ret_operation(rets)
            else:
                self.classification_ret_operation(rets)
            self.init_buffer()

    def feed_data_forced(self):
        if self._buffer_length == 0:
            return
        rets = self.model.get_data(self._data_buffer['data'])
        if self._model_type == NGRAM_TYPE:
            self.ngram_ret_operation(rets)
        elif self._model_type == RNN_TYPE:
            self.nn_ret_operation(rets)
        elif self._model_type == PROB_MODEL_TYPE or self._model_type == PROB_MODEL_NGRAM_TYPE:
            self.prob_ret_operation(rets)
        else:
            self.classification_ret_operation(rets)
        self.init_buffer()

    def feed_main_data(self, main_data, main_data_ID):
        self._main_data_dict[main_data_ID] = main_data

    def feed_tokens_data(self, token_contents, data_ID_tuple):
        if self._model_type == PROB_MODEL_NGRAM_TYPE:
            rets = self.model_2.score(token_contents)
            self._main_token_rets[str(data_ID_tuple)] = rets

    def generate_prob_ngram_rets_log(self, data_ID_tuple):
        rets = self._main_token_rets[str(data_ID_tuple)]
        log_str = "\nNGRAM_RETS: "
        for ret in rets:
            log_str += "%s[%d, %d, %d] " % (ret[0], ret[1][0], ret[1][1], ret[1][2])
        log_str += "\n\n"
        return log_str

    # data_ID_tuple: [main_data_ID, correct_or_wrong]
    def log_data(self, log_str, data_ID_tuple):
        if data_ID_tuple != self._data_ID_tuple:
            if self._test_mode == 1:
                self._log_handle.write("\n\n\n#SENTENCE: %s \n" % self._main_data_dict[data_ID_tuple[0]])
            else:
                # if data_ID_tuple[1] == 0:
                #     sentence = self._main_data_dict[data_ID_tuple[0]]["correct_sentence"]
                # else:
                #     sentence = self._main_data_dict[data_ID_tuple[0]]["wrong_sentence"]
                # self._log_handle.write("\n\n\n#SENTENCE: %s \n" % sentence)
                self._log_handle.write(generate_log_sentence(self._main_data_dict[data_ID_tuple[0]], data_ID_tuple[1]))
                if self._model_type == PROB_MODEL_NGRAM_TYPE:
                    self._log_handle.write(self.generate_prob_ngram_rets_log(data_ID_tuple))
            self._data_ID_tuple = data_ID_tuple
        self._log_handle.write(log_str)

    def ngram_ret_operation(self, rets):
        for data_ID_tuple, data, scores in zip(self._data_buffer['data_ID_tuple'], self._data_buffer['data'], rets):
            sum_score = 0
            for score_info in scores:
                sum_score += score_info[0]
            if data[0] == 0:
                type_word = 'D'
            else:
                type_word = 'B'
            if len(scores) == 6:
                log_str = "%s: %s [%.2f, %d] %s [%.2f, %d] %s [%.2f, %d] %s [%.2f, %d] %s [%.2f, %d] # %.2f \n" % (
                    type_word, data[1], scores[0][0], scores[0][1], data[2], scores[1][0], scores[1][1],
                    data[3], scores[2][0], scores[2][1], data[4], scores[3][0], scores[3][1], data[5],
                    scores[4][0], scores[4][1], sum_score)
            else:
                log_str = "%s: %s [%.2f, %d] %s [%.2f, %d] %s [%.2f, %d] # %.2f \n" % (
                    type_word, data[1], scores[0][0], scores[0][1], data[2], scores[1][0], scores[1][1],
                    data[3], scores[2][0], scores[2][1], sum_score)
            self.log_data(log_str, data_ID_tuple)


    # NEED TO BE MODIFIED
    def nn_ret_operation(self, rets):
        for data_ID_tuple, data, scores in zip(self._data_buffer['data_ID_tuple'], self._data_buffer['data'], rets):
            sum_score = 0
            oov_flag = False
            print_flag = False
            min_score = 100
            debug_score_list = [self._debug_model.score(data[i + 2], bos=False, eos=False) if score_info[0] < -6 else 0
                                for i, score_info in zip(range(len(scores)), scores)]
            # for score_info in scores:
            #     if score_info[1]:
            #         oov_flag = True
            #     else:
            #         if score_info[0] < -10:
            #             print_flag = True
            #         if score_info[0] < min_score:
            #             min_score = score_info[0]
            #     sum_score += score_info[0]
            for score_info, debug_score in zip(scores, debug_score_list):
                if score_info[1]:
                    oov_flag = True
                else:
                    if score_info[0] < -6 and debug_score > -5.8:
                        print_flag = True
                        if score_info[0] < min_score:
                            min_score = score_info[0]
                sum_score += score_info[0]
            if oov_flag:
                continue
            if data[0] == 0:
                type_word = 'D'
            else:
                type_word = 'B'

            log_str = "%s: %s %s [%.2f, %d, %.2f] %s [%.2f, %d, %.2f] %s [%.2f, %d, %.2f] %s [%.2f, %d, %.2f] # %.2f \n" % (
                type_word, data[1], data[2], scores[0][0], scores[0][1], debug_score_list[0],
                data[3], scores[1][0], scores[1][1], debug_score_list[1], data[4], scores[2][0], scores[2][1],
                debug_score_list[2], data[5],
                scores[3][0], scores[3][1], debug_score_list[3], sum_score)

            # log_str = "%s: %s %s [%.2f, %d] %s [%.2f, %d] %s [%.2f, %d] %s [%.2f, %d] # %.2f \n" % (
            #     type_word, data[1], data[2], scores[0][0], scores[0][1],
            #     data[3], scores[1][0], scores[1][1], data[4], scores[2][0], scores[2][1], data[5],
            #     scores[3][0], scores[3][1], sum_score)
            if sum_score < 0 and print_flag:
                self.log_data(log_str, data_ID_tuple)


    # NEED TO BE MODIFIED
    def classification_ret_operation(self, rets):
        for data_ID_tuple, data, ret in zip(self._data_buffer['data_ID_tuple'], self._data_buffer['data'], rets):
            if data[0] == 0:
                type_word = 'D'
            else:
                type_word = 'B'
            scores = ret[0]
            if scores[0] > scores[1]:
                result = 0
            else:
                result = 1
            log_str = "%s: %s %s %s %s %s # %d          %s,    %s\n" % (
                type_word, data[1], data[2], data[3], data[4], data[5], result, str(ret[0]), str(ret[1]))
            self.log_data(log_str, data_ID_tuple)

    def prob_ret_operation(self, rets):
        for data_ID_tuple, data, ret in zip(self._data_buffer['data_ID_tuple'], self._data_buffer['data'], rets):
            if data[0] == 0:
                type_word = 'D'
            else:
                type_word = 'B'
            word_1_info = data[1]
            word_2_info = data[3]
            word_3_info = data[5]
            label_1 = data[2]
            label_2 = data[4]
            word_1 = word_1_info["text"]
            word_2 = word_2_info["text"]
            word_3 = word_3_info["text"]

            log_str = "%s: %s-%s-%d %s %s-%s-%d %s %s-%s-%d #\n" % (
                type_word, word_1_info['text'], word_1_info['POS'], word_1_info['index'][0], label_1,
                word_2_info['text'], word_2_info['POS'], word_2_info['index'][0], label_2, word_3_info['text'],
                word_3_info['POS'], word_3_info['index'][0])
            for model_type in range(len(ret)):
                if model_type == 0:
                    log_str += "%s\t\t\t[%d]; \n" % (word_1, ret[model_type])
                elif model_type == 1:
                    log_str += "%s\t\t\t[%d]; \n" % (word_2, ret[model_type])
                elif model_type == 2:
                    log_str += "%s\t\t\t[%d]; \n" % (word_3, ret[model_type])
                elif model_type == 3:
                    log_str += "%s\tX\t--\t[%d]; \n" % (word_1, ret[model_type])
                elif model_type == 4:
                    log_str += "X\t%s\t--\t[%d]; \n" % (word_2, ret[model_type])
                elif model_type == 5:
                    log_str += "%s\t%s\t--\t[%d]; \n" % (word_1, word_2, ret[model_type])
                elif model_type == 6:
                    if data[0] == 0:
                        log_str += "%s\tX\t--\t[%d]; \n" % (word_2, ret[model_type])
                    else:
                        log_str += "%s\tX\t--\t[%d]; \n" % (word_1, ret[model_type])
                elif model_type == 7:
                    log_str += "X\t%s\t--\t[%d]; \n" % (word_3, ret[model_type])
                elif model_type == 8:
                    if data[0] == 0:
                        log_str += "%s\t%s\t--\t[%d]; \n" % (word_2, word_3, ret[model_type])
                    else:
                        log_str += "%s\t%s\t--\t[%d]; \n" % (word_1, word_3, ret[model_type])
                elif model_type == 9:
                    log_str += "%s\tX\tX\t[%d]; \n" % (word_1, ret[model_type])
                elif model_type == 10:
                    log_str += "X\t%s\tX\t[%d]; \n" % (word_2, ret[model_type])
                elif model_type == 11:
                    log_str += "X\tX\t%s\t[%d]; \n" % (word_3, ret[model_type])
                elif model_type == 12:
                    log_str += "%s\t%s\tX\t[%d]; \n" % (word_1, word_2, ret[model_type])
                elif model_type == 13:
                    log_str += "%s\tX\t%s\t[%d]; \n" % (word_1, word_3, ret[model_type])
                elif model_type == 14:
                    log_str += "X\t%s\t%s\t[%d]; \n" % (word_2, word_3, ret[model_type])
                elif model_type == 15:
                    log_str += "%s\t%s\t%s\t[%d]; \n" % (word_1, word_2, word_3, ret[model_type])
                elif model_type == 16:
                    log_str += "%s*\t%s\t%s\t[%d]*; \n" % (word_1, word_2, word_3, ret[model_type])
                elif model_type == 17:
                    log_str += "%s\t%s*\t%s\t[%d]*; \n" % (word_1, word_2, word_3, ret[model_type])
                elif model_type == 18:
                    log_str += "%s\t%s\t%s*\t[%d]*; \n\n" % (word_1, word_2, word_3, ret[model_type])
            self.log_data(log_str, data_ID_tuple)

    def transform_scores_to_str(self, rets):
        ret_str = ""
        if self._model_type == CLASSIFICATION_TYPE:
            for ret in rets:
                ret_str += str(ret[0][1]) + ", "
        else:
            for ret in rets:
                ret_str += str(ret) + ", "
        return ret_str

