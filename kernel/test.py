import argparse
from .ngram import Ngram
from .reader import Reader
from . import kernel
import json
import os

RECORD_NUM = 50
THRESHOLDS = (-20, -8)


def analyze_sentence(sentence):
    ret = kernel.get_parsed_ret(sentence)
    tokens = kernel.get_tokens(ret)
    dependency_tree = kernel.build_dependency_tree(ret, tokens)
    log_handle.write("#Sentence: %s\n" % sentence)
    scores = kernel.parse_tree(dependency_tree, tokens, log_handle)
    return scores


def transform_scores_to_str(scores):
    ret = ""
    for score in scores:
        ret += str(score) + " "
    return ret


def save_to_csv(data, wrong_scores, correct_scores):
    wrong_sentence = data["wrong_sentence"]
    correct_sentence = data["correct_sentence"]
    mistake_type = data["mistake_type"]
    wrong_word = data["wrong_word"]
    correct_word = data["correct_word"]
    word_position = str(data["word_position"])
    correct_scores_str = transform_scores_to_str(correct_scores)
    wrong_scores_str = transform_scores_to_str(wrong_scores)
    csv_handle.write(
        '"%s", "%s", "%s", "%s", %s\n' % (mistake_type, wrong_sentence, wrong_word, word_position, wrong_scores_str))
    csv_handle.write(
        '"%s", "%s", "%s", "%s", %s\n' % ("0", correct_sentence, correct_word, word_position, correct_scores_str))


def update_precisions(data_flag, scores):
    delta = float(THRESHOLDS[1] - THRESHOLDS[0]) / float(RECORD_NUM)
    min_score = min(scores)
    threshold = THRESHOLDS[0] - delta
    for i in range(RECORD_NUM):
        threshold += delta
        if min_score < threshold:
            if data_flag == 1:
                correct_nums_1[i] += 1
        if min_score > threshold:
            if data_flag == 0:
                correct_nums_0[i] += 1


def operate_data(data):
    wrong_scores = analyze_sentence(data["wrong_sentence"])
    correct_scores = analyze_sentence(data["correct_sentence"])
    save_to_csv(data, wrong_scores, correct_scores)
    update_precisions(1, wrong_scores)
    update_precisions(0, correct_scores)


def get_max_correct_nums():
    max_correct_num = 0
    best_No = 0
    for i in range(RECORD_NUM):
        correct_num = correct_nums_0[i] + correct_nums_1[i]
        if correct_num > max_correct_num:
            best_No = i
            max_correct_num = correct_num
    return (correct_nums_0[best_No], correct_nums_1[best_No])


def save_precisions(operated_num, precision_log_path):
    output_str = ""
    for num in correct_nums_0:
        precision = float(num) / operated_num
        output_str += str(precision) + ", "
    output_str += "\n"
    for num in correct_nums_1:
        precision = float(num) / operated_num
        output_str += str(precision) + ", "
    with open(precision_log_path, 'w') as precision_log_handle:
        precision_log_handle.write(output_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('--output_dir', type=str, default="./ret")
    parser.add_argument('--log_name', type=str, default='log.txt')
    parser.add_argument('--model_flag', type=int, default=0)
    parser.add_argument('--statistics_name', type=str, default="statistics.csv")
    parser.add_argument('--precision_log_name', type=str, default="precisions.csv")
    args = parser.parse_args()
    log_name = args.log_name
    output_dir = args.output_dir
    input_path = args.input_path
    model_flag = args.model_flag
    statistics_name = args.statistics_name
    precision_log_name = args.precision_log_name

    kernel.ngram_model = Ngram(model_flag)
    file_num = 0
    if not input_path.endswith('.out') and not input_path.endswith('.txt'):
        exit()

    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except:
            print("Output directory is invalid!")
            exit()

    log_path = os.path.join(output_dir, log_name)
    precision_log_path = os.path.join(output_dir, precision_log_name)
    statistics_path = os.path.join(output_dir, statistics_name)

    reader = Reader(input_path)

    global log_handle
    log_handle = open(log_path, 'w')
    global csv_handle
    csv_handle = open(statistics_path, 'w')

    operated_num = 0
    global correct_nums_0
    global correct_nums_1
    correct_nums_0 = [0 for i in range(RECORD_NUM)]
    correct_nums_1 = [0 for i in range(RECORD_NUM)]
    for data in reader():
        if operated_num % 100 == 0 and operated_num != 0:
            max_correct_num = get_max_correct_nums()
            max_ratio_1 = max_correct_num[1] / float(operated_num)
            max_ratio_0 = max_correct_num[0] / float(operated_num)
            print("Operated Sentences Num: %s; best_correct_ratio: (%f, %f)" % (operated_num, max_ratio_0, max_ratio_1),
                  end="\r")
        operate_data(data)
        operated_num += 1
    save_precisions(operated_num, precision_log_path)
    correct_nums_json = json.dumps([correct_nums_0, correct_nums_1])
    print(correct_nums_json)
    log_handle.close()
