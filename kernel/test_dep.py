import argparse
import json
import os

from . import kernel
from .reader import Reader
from .data_container import Data_container

PARSING_BUFFER_SIZE = 100

def analyze_sentence(sentence_ID_tuple, ret):
    tokens, token_contents = kernel.get_tokens(ret)
    data_container.feed_tokens_data(token_contents, sentence_ID_tuple)
    dependency_tree = kernel.build_dependency_tree(ret, tokens)
    kernel.parse_tree(dependency_tree, tokens, sentence_ID_tuple, data_container)


def operate_data(data, data_No):
    wrong_ret = kernel.get_parsed_ret(data["wrong_sentence"])
    if not wrong_ret:
        return False
    correct_ret = kernel.get_parsed_ret(data["correct_sentence"])
    if not correct_ret:
        return False
    data_container.feed_main_data(data, data_No)
    analyze_sentence((data_No, 1), wrong_ret)
    analyze_sentence((data_No, 0), correct_ret)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('--output_dir', type=str, default="./ret")
    parser.add_argument('--log_name', type=str, default='log.txt')
    parser.add_argument('--model_flag', type=int, default=0)
    parser.add_argument('--statistics_name', type=str, default="statistics.csv")
    parser.add_argument('--precision_log_name', type=str, default="precisions.csv")
    parser.add_argument('--data_type', type=int, default=0)

    args = parser.parse_args()
    log_name = args.log_name
    output_dir = args.output_dir
    input_path = args.input_path
    model_flag = args.model_flag
    data_type = args.data_type
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
    elif model_flag == 2:
        data_container = Data_container(log_path=log_path, precisions_path=precision_log_path,
                                        statistics_path=statistics_path, model_type='prob_model', buffer_size=1)
    elif model_flag == 3:
        data_container = Data_container(log_path=log_path, precisions_path=precision_log_path,
                                        statistics_path=statistics_path, model_type='prob_model_ngram', buffer_size=1)
    else:
        data_container = Data_container(log_path=log_path, precisions_path=precision_log_path,
                                        statistics_path=statistics_path, model_type='classification', buffer_size=256)
    reader = Reader(input_path, data_type)
    data_No = 0
    buffer_index = 0
    data_No_buffer = []
    sentence_buffer = []
    for data in reader():
        sentence_buffer.append(data["wrong_sentence"])
        sentence_buffer.append(data["correct_sentence"])
        data_No_buffer.append((data_No, 1))
        data_No_buffer.append((data_No, 0))
        buffer_index += 2
        data_No += 1
        if buffer_index >= PARSING_BUFFER_SIZE:
            para = "\n".join(sentence_buffer)
            rets = kernel.get_parsed_rets(para)
            if not rets or len(rets) != len(sentence_buffer):
                print(para)
                sentence_buffer = []
                data_No_buffer = []
                buffer_index = 0
                continue
            for ret, data_No_tuple in zip(rets, data_No_buffer):
                if data_No_tuple[1] == 1:
                    data_container.feed_main_data(data, data_No_tuple[0])
                analyze_sentence(data_No_tuple, ret)
            sentence_buffer = []
            data_No_buffer = []
            buffer_index = 0
            print("Having Finished: %d" % data_No, end='\r')
    data_container.feed_data_forced()
    # data_container.dump_csv_statistics()
    # data_container.dump_csv_precision()
    data_container.close_all_handles()
