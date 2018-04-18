import argparse
import json
import os

from . import kernel
from .reader import Reader
from .data_container import Data_container


def analyze_sentence(sentence, sentence_ID):
    ret = kernel.get_parsed_ret(sentence)
    tokens = kernel.get_tokens(ret)
    dependency_tree = kernel.build_dependency_tree(ret, tokens)
    kernel.parse_tree(dependency_tree, tokens, sentence_ID, data_container)


def operate_data(data, data_No):
    data_container.feed_main_data(data, data_No)
    analyze_sentence(data["wrong_sentence"], (data_No, 1))
    analyze_sentence(data["correct_sentence"], (data_No, 0))


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

    global data_container
    if model_flag == 0:
        data_container = Data_container(log_path=log_path, precisions_path=precision_log_path,
                                        statistics_path=statistics_path, model_type='ngram')
    elif model_flag == 1:
        data_container = Data_container(log_path=log_path, precisions_path=precision_log_path,
                                        statistics_path=statistics_path, model_type='rnn', buffer_size=200)
    else:
        data_container = Data_container(log_path=log_path, precisions_path=precision_log_path,
                                        statistics_path=statistics_path, model_type='classification', buffer_size=200)
    reader = Reader(input_path)
    data_No = 0

    for data in reader():
        if data_No % 100 == 0 and data_No != 0:
            print("Having Finished: %d" % data_No, end='\r')
        operate_data(data, data_No)
        data_No += 1
    data_container.feed_data_forced()
    data_container.dump_csv_statistics()
    data_container.dump_csv_precision()
    data_container.close_all_handles()
