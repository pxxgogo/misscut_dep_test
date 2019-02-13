import argparse
import os
import re

from . import kernel
from .reader import Reader
from .data_container import Data_container

PARSING_BUFFER_SIZE = 100
CUT_FLAG_REG = re.compile('[，,。！!？\?……：:；;\n\r —\.]+')


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


def check_data(data):
    sentence = data["wrong_sentence"]
    items = CUT_FLAG_REG.split(sentence)
    ignorant_num = 0
    if items[-1] == "":
        ignorant_num = 1
    size = len(items) - ignorant_num
    if size != 1:
        return False
    sentence = data["correct_sentence"]
    items = CUT_FLAG_REG.split(sentence)
    ignorant_num = 0
    if items[-1] == "":
        ignorant_num = 1
    size = len(items) - ignorant_num
    if size != 1:
        return False
    return True


# gpu_flag is for smooth algorithm
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('--output_dir', type=str, default="./ret")
    parser.add_argument('--log_name', type=str, default='log.txt')
    parser.add_argument('--model_flag', type=int, default=0)
    parser.add_argument('--data_type', type=int, default=0)
    parser.add_argument('--c_w_data_chosen_flag', type=int, default=0)
    parser.add_argument('--max_data_num', type=int, default=-1)
    parser.add_argument('--gpu_flag', type=int, default=0)

    args = parser.parse_args()
    log_name = args.log_name
    output_dir = args.output_dir
    input_path = args.input_path
    model_flag = args.model_flag
    data_type = args.data_type
    c_w_data_chosen_flag = args.c_w_data_chosen_flag
    max_data_num = args.max_data_num
    gpu_flag = args.gpu_flag
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

    global data_container
    if model_flag == 0:
        data_container = Data_container(log_path=log_path, model_type='ngram', gpu_flag=gpu_flag)
    elif model_flag == 1:
        data_container = Data_container(log_path=log_path, model_type='rnn', buffer_size=200, gpu_flag=gpu_flag)
    elif model_flag == 2:
        data_container = Data_container(log_path=log_path, model_type='prob_model', buffer_size=1, gpu_flag=gpu_flag)
    elif model_flag == 3:
        data_container = Data_container(log_path=log_path, model_type='prob_model_ngram', buffer_size=1, gpu_flag=gpu_flag)
    else:
        data_container = Data_container(log_path=log_path, model_type='classification', buffer_size=256, gpu_flag=gpu_flag)
    reader = Reader(input_path, data_type)
    data_No = 0
    buffer_index = 0
    data_No_buffer = []
    sentence_buffer = []
    data_buffer = []
    pre_data_No = -1

    for data in reader():
        if not check_data(data):
            continue
        if c_w_data_chosen_flag in [0, 2]:
            sentence_buffer.append(data["wrong_sentence"])
            data_No_buffer.append((data_No, 1))
            data_buffer.append(data)
            buffer_index += 1

        if c_w_data_chosen_flag in [0, 1]:
            sentence_buffer.append(data["correct_sentence"])
            data_No_buffer.append((data_No, 0))
            data_buffer.append(data)
            buffer_index += 1

        data_No += 1
        if buffer_index >= PARSING_BUFFER_SIZE:
            para = "\n".join(sentence_buffer)
            rets = kernel.get_parsed_rets(para)
            if not rets:
                print(para)
                sentence_buffer = []
                data_No_buffer = []
                data_buffer = []
                buffer_index = 0
                continue
            if len(rets) != len(sentence_buffer):
                print(len(rets), len(sentence_buffer))
                sentence_buffer = []
                data_No_buffer = []
                data_buffer = []
                buffer_index = 0
                continue
            for ret, data_No_tuple, key_data in zip(rets, data_No_buffer, data_buffer):
                if data_No_tuple[0] != pre_data_No:
                    data_container.feed_main_data(key_data, data_No_tuple[0])
                    pre_data_No = data_No_buffer[0]
                analyze_sentence(data_No_tuple, ret)
            sentence_buffer = []
            data_No_buffer = []
            data_buffer = []
            buffer_index = 0
            print("Having Finished: %d" % data_No, end='\r')
            if 0 <= max_data_num < data_No:
                break

    data_container.feed_data_forced()
    data_container.close_all_handles()
