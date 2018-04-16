import re

POSITION_COMPILER = re.compile("\[(\d+),(\d+)\]")

IGNORE_TYPES = ["3-1", "3-2"]

class Reader:
    def __init__(self, file_path):
        with open(file_path) as file_handle:
            raw_datas = file_handle.readlines()
            self.parse_data(raw_datas)

    def parse_data(self, raw_datas):
        self.__datas = []
        # print(raw_datas)
        for line in raw_datas:
            items = line.strip().split("\t")
            # print(items)
            if len(items) < 7:
                continue
            wrong_sentence = items[0]
            mistake_type = items[1]
            if mistake_type in IGNORE_TYPES:
                continue
            wrong_word = items[2]
            correct_word = items[4]
            word_position_str = POSITION_COMPILER.findall(items[5])[0]
            word_position = (int(word_position_str[0]), int(word_position_str[1]))
            mistake_flag = items[6]
            if mistake_flag == 'False':
                continue
            correct_sentence = wrong_sentence[:word_position[0]] + correct_word + wrong_sentence[word_position[1]:]
            data = {'correct_sentence': correct_sentence,
                    'wrong_sentence': wrong_sentence,
                    'mistake_type': mistake_type,
                    'wrong_word': wrong_word,
                    'correct_word': correct_word,
                    'word_position': word_position}
            self.__datas.append(data)

    def __call__(self):
        for data in self.__datas:
            yield data


if __name__ == "__main__":
    reader = Reader("mini_test_corpus.txt")
    for data in reader():
        print(data)
        input()











            
