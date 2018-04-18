import re
from .model.ngram import ngram
from .model.rnn import wrapper

NGRAM_TYPE = 'ngram'
RNN_TYPE = 'rnn'
CLASSIFICATION = "classification"
RECORD_NUM = 50
THRESHOLDS = (-20, -8)

DATA_ID_TUPLE_COMPILER = re.compile("\((\d+), (\d+)\)")


class Data_container:
    def __init__(self, buffer_size=1, model_type=NGRAM_TYPE, log_path="./log.txt", statistics_path="./statistics.csv",
                 precisions_path="precisions.csv", test_mode=0):
        self._data_buffer = {'data_ID_tuple': [], 'data': []}
        self._buffer_length = 0
        self._scores = {}
        self._log_handle = open(log_path, 'w')
        self._statistics_handle = open(statistics_path, 'w')
        self._precisions_handle = open(precisions_path, 'w')
        self._buffer_size = buffer_size
        self._model_type = model_type
        self._main_data_dict = {}
        self._test_mode = test_mode
        self._data_ID_tuple = -1
        if model_type == NGRAM_TYPE:
            self.model = ngram.Ngram()
        else:
            self.model = wrapper.RNN_wrapper()

    def init_buffer(self):
        self._data_buffer = {'data_ID_tuple': [], 'data': []}
        self._buffer_length = 0

    def close_all_handles(self):
        self._log_handle.close()
        self._statistics_handle.close()
        self._precisions_handle.close()

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
        else:
            self.classification_ret_operation(rets)
        self.init_buffer()

    def feed_main_data(self, main_data, main_data_ID):
        self._main_data_dict[main_data_ID] = main_data

    # data_ID_tuple: [main_data_ID, correct_or_wrong]
    def log_data(self, log_str, data_ID_tuple):
        if data_ID_tuple != self._data_ID_tuple:
            if self._test_mode == 1:
                self._log_handle.write("#SENTENCE: %s \n" % self._main_data_dict[data_ID_tuple[0]])
            else:
                if data_ID_tuple[1] == 0:
                    sentence = self._main_data_dict[data_ID_tuple[0]]["correct_sentence"]
                else:
                    sentence = self._main_data_dict[data_ID_tuple[0]]["wrong_sentence"]
                self._log_handle.write("#SENTENCE: %s \n" % sentence)
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
            data_ID_tuple_str = str(data_ID_tuple)
            scores_per_sentence = self._scores.get(data_ID_tuple_str, [])
            scores_per_sentence.append(sum_score)
            self._scores[data_ID_tuple_str] = scores_per_sentence

    def nn_ret_operation(self, rets):
        for data_ID_tuple, data, scores in zip(self._data_buffer['data_ID_tuple'], self._data_buffer['data'], rets):
            sum_score = 0
            oov_flag = False
            print_flag = False
            min_score = 100
            for score_info in scores:
                if score_info[1]:
                    oov_flag = True
                else:
                    if score_info[0] < -10:
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

            log_str = "%s: %s %s [%.2f, %d] %s [%.2f, %d] %s [%.2f, %d] %s [%.2f, %d] # %.2f \n" % (
                type_word, data[1], data[2], scores[0][0], scores[0][1],
                data[3], scores[1][0], scores[1][1], data[4], scores[2][0], scores[2][1], data[5],
                scores[3][0], scores[3][1], sum_score)
            if sum_score < 0 and print_flag:
                self.log_data(log_str, data_ID_tuple)
            data_ID_tuple_str = str(data_ID_tuple)
            scores_per_sentence = self._scores.get(data_ID_tuple_str, [])
            scores_per_sentence.append(min_score)
            self._scores[data_ID_tuple_str] = scores_per_sentence

    def classification_ret_operation(self, rets):
        for data_ID_tuple, data, ret in zip(self._data_buffer['data_ID_tuple'], self._data_buffer['data'], rets):
            if data[0] == 0:
                type_word = 'D'
            else:
                type_word = 'B'
            if ret[0] > ret[1]:
                result = 0
            else:
                result = 1
            log_str = "%s: %s %s %s %s %s # %d \n" % (type_word, data[1], data[2], data[3], data[4], data[5], result)
            self.log_data(log_str, data_ID_tuple)
            data_ID_tuple_str = str(data_ID_tuple)
            scores_per_sentence = self._scores.get(data_ID_tuple_str, [])
            scores_per_sentence.append(ret)
            self._scores[data_ID_tuple_str] = scores_per_sentence

    def transform_scores_to_str(self, rets):
        ret_str = ""
        if self._model_type == CLASSIFICATION:
            for ret in rets:
                ret_str += str(ret[1]) + ", "
        else:
            for ret in rets:
                ret_str += str(ret) + ", "
        return ret_str

    def dump_csv_statistics(self):
        if self._test_mode == 1:
            return
        for main_data_id, main_data in self._main_data_dict.items():
            wrong_sentence = main_data["wrong_sentence"]
            correct_sentence = main_data["correct_sentence"]
            mistake_type = main_data["mistake_type"]
            wrong_word = main_data["wrong_word"]
            correct_word = main_data["correct_word"]
            word_position = str(main_data["word_position"])
            correct_scores_str = self.transform_scores_to_str(self._scores.get(str((main_data_id, 0)), []))
            wrong_scores_str = self.transform_scores_to_str(self._scores.get(str((main_data_id, 1)), []))
            self._statistics_handle.write(
                '"%s", "%s", "%s", "%s", %s\n' % (
                    mistake_type, wrong_sentence, wrong_word, word_position, wrong_scores_str))
            self._statistics_handle.write(
                '"%s", "%s", "%s", "%s", %s\n' % (
                    "0", correct_sentence, correct_word, word_position, correct_scores_str))

    def dump_csv_precision(self):
        if self._model_type == CLASSIFICATION:
            return
        correct_nums = [[0 for i in range(RECORD_NUM)], [0 for i in range(RECORD_NUM)]]
        delta = float(THRESHOLDS[1] - THRESHOLDS[0]) / float(RECORD_NUM)
        for data_ID_tuple_str, scores in self._scores.items():
            data_ID_tuple = DATA_ID_TUPLE_COMPILER.findall(data_ID_tuple_str)[0]
            data_flag = int(data_ID_tuple[1])
            min_score = min(scores)
            threshold = THRESHOLDS[0] - delta
            for i in range(RECORD_NUM):
                threshold += delta
                if min_score < threshold:
                    if data_flag == 1:
                        correct_nums[1][i] += 1
                else:
                    if data_flag == 0:
                        correct_nums[0][i] += 1
        output_str = ""
        main_data_num = len(self._main_data_dict)
        for num in correct_nums[0]:
            precision = float(num) / main_data_num
            output_str += str(precision) + ", "
        output_str += "\n"
        for num in correct_nums[1]:
            precision = float(num) / main_data_num
            output_str += str(precision) + ", "
        self._precisions_handle.write(output_str)
