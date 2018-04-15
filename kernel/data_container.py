from . import ngram

NGRAM_TYPE = 'ngram'
RNN_TYPE = 'rnn'


class Data_container:
    def __init__(self, buffer_size=1, model_type="ngram", log_path="./log.txt"):
        self._data_buffer = {'sentence_Nos': [], 'data': []}
        self._buffer_length = 0
        self._scores = {}
        self._log_handle = open(log_path, 'w')
        self._buffer_size = buffer_size
        self._model_type = model_type
        self._sentence_dict = {}
        self._log_sentence_No = -1
        if model_type == NGRAM_TYPE:
            self.model = ngram.Ngram()

    def init_buffer(self):
        self._data_buffer = {'sentence_Nos': [], 'data': []}
        self._buffer_length = 0

    def close_log_handle(self):
        self._log_handle.close()

    def feed_data(self, data, sentence_No, end_flag=False):
        self._data_buffer['sentence_Nos'].append(sentence_No)
        self._data_buffer['data'].append(data)
        self._buffer_length += 1
        if self._buffer_length == self._buffer_size or end_flag:
            rets = self.model.get_data(self._data_buffer['data'])
            if self._model_type == NGRAM_TYPE:
                self.ngram_ret_operation(rets)
            else:
                self.nn_ret_operation(rets)
            self.init_buffer()

    def feed_sentence(self, sentence, sentence_No):
        self._sentence_dict[sentence_No] = sentence

    def log_data(self, log_str, sentence_No):
        if sentence_No != self._log_sentence_No:
            self._log_handle.write("#SENTENCE: %s \n" % self._sentence_dict[sentence_No])
            self._log_sentence_No = sentence_No
        self._log_handle.write(log_str)

    def ngram_ret_operation(self, rets):
        for sentence_No, data, scores in zip(self._data_buffer['sentence_Nos'], self._data_buffer['data'], rets):
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
            self.log_data(log_str, sentence_No)
            scores_per_sentence = self._scores.get(sentence_No, [])
            scores_per_sentence.append(scores)
            self._scores[sentence_No] = scores_per_sentence

    def nn_ret_operation(self, rets):
        for sentence_No, data, scores in zip(self._data_buffer['sentence_Nos'], self._data_buffer['data'], rets):
            sum_score = 0
            for score in scores:
                sum_score += score
            if data[0] == 0:
                type_word = 'D'
            else:
                type_word = 'B'
            if len(scores) == 6:
                log_str = "%s: %s [%.2f] %s [%.2f] %s [%.2f] %s [%.2f] %s [%.2f] # %.2f \n" % (
                    type_word, data[1], scores[0], data[2], scores[1],
                    data[3], scores[2], data[4], scores[3], data[5],
                    scores[4], sum_score)
            else:
                log_str = "%s: %s [%.2f] %s [%.2f] %s [%.2f] # %.2f \n" % (
                    type_word, data[1], scores[0], data[2], scores[1],
                    data[3], scores[2], sum_score)
            self.log_data(log_str, sentence_No)
            scores_per_sentence = self._scores.get(sentence_No, [])
            scores_per_sentence.append(scores)
            self._scores[sentence_No] = scores_per_sentence


