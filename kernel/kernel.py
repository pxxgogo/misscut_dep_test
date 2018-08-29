import argparse
import re

import requests

from .data_container import Data_container

CORENLP_SERVER_URL = 'http://166.111.139.15:9000'

HEADER_PATTERN = r'Sentence #(\d+) \((\d+) tokens\):'
HEADER_COMPILER = re.compile(HEADER_PATTERN)

TOKEN_PATTERN = r'\[Text=(\S+) CharacterOffsetBegin=(\d+) CharacterOffsetEnd=(\d+) PartOfSpeech=(\w+)\]'
TOKEN_COMPILER = re.compile(TOKEN_PATTERN)

DEPENDENCY_HEADER_PREFIX = 'Dependency Parse'

DEPENDENCY_ITEM_PATTERN = r'(\w+)\((\S+)-(\d+), (\S+)-(\d+)\)'
DEPENDENCY_ITEM_COMPILER = re.compile(DEPENDENCY_ITEM_PATTERN)
SENTENCE_MAX_LENGTH = 50


def search_beginning(lines, line_No, lines_num):
    while line_No < lines_num:
        header = lines[line_No].strip()
        header_ret_list = HEADER_COMPILER.findall(header)
        if len(header_ret_list) != 0:
            return line_No, header_ret_list
        line_No += 1
    return -1, None


def operate_sub_tree(type, word_0_info, label_0, word_1_info, label_1, word_2_info, sentence_No, data_container):
    data = (type, word_0_info, label_0, word_1_info, label_1, word_2_info)
    data_container.feed_data(data, sentence_No)


def search_pattern_1(node_No, tokens, tree, sentence_No, data_container):
    node_info = tree[node_No]
    for child_info in node_info:
        for grandchild_info in tree[child_info['child']]:
            operate_sub_tree(0, tokens[node_No], child_info['relation'],
                             tokens[child_info['child']], grandchild_info['relation'],
                             tokens[grandchild_info['child']], sentence_No, data_container)


def search_pattern_2(node_No, tokens, tree, sentence_No, data_container):
    node_info = tree[node_No]
    child_num = len(node_info)
    for child_1_No in range(child_num):
        for child_2_No in range(child_1_No + 1, child_num):
            child_1_info = node_info[child_1_No]
            child_2_info = node_info[child_2_No]
            operate_sub_tree(1, tokens[node_No], child_1_info['relation'],
                             tokens[child_1_info['child']], child_2_info['relation'],
                             tokens[child_2_info['child']], sentence_No, data_container)


def parse_node(node_No, tokens, tree, sentence_No, data_container):
    search_pattern_1(node_No, tokens, tree, sentence_No, data_container)
    search_pattern_2(node_No, tokens, tree, sentence_No, data_container)
    for child_info in tree[node_No]:
        parse_node(child_info['child'], tokens, tree, sentence_No, data_container)


def parse_tree(tree, tokens, sentence_No, data_container):
    parse_node(0, tokens, tree, sentence_No, data_container)


def get_tokens(ret_json):
    tokens = []
    tokens.append({'text': '{ROOT}', 'index': (-1, -1), 'POS': 'ROOT'})
    tokens_json = ret_json["tokens"]
    tokens_content = []
    for token_info in tokens_json:
        tokens.append({'text': token_info["originalText"],
                       'index': (token_info["characterOffsetBegin"], token_info["characterOffsetEnd"]),
                       'POS': token_info["pos"]})
        tokens_content.append(token_info["originalText"])
    return tokens, tokens_content


def find_line_No_by_prefix(lines, pattern, line_No):
    line_num = len(lines)
    while True:
        if line_No >= line_num:
            return -1
        line = lines[line_No].strip()
        if line.startswith(pattern):
            return line_No
        line_No += 1


def get_simple_relation(relation):
    critical_position = relation.find(":")
    while critical_position != -1:
        relation = relation[critical_position + 1:]
        critical_position = relation.find(":")
    return relation


def build_dependency_tree(ret_json, tokens):
    dependency_tree = [[] for i in range(len(tokens))]
    ret_dep_info = ret_json["enhancedPlusPlusDependencies"]
    for dep_info in ret_dep_info:
        father_node_No = dep_info['governor']
        child_node_No = dep_info['dependent']
        relation = dep_info['dep'].lower()
        dependency_tree[father_node_No].append({'relation': get_simple_relation(relation), 'child': child_node_No})
    return dependency_tree


def get_parsed_ret(para):
    properties = '{"annotators":"tokenize,ssplit,pos,depparse","outputFormat":"json","ssplit.newlineIsSentenceBreak":"always"}'
    # print(line)
    ret = requests.post(data=para.encode("utf-8"), params={"properties": properties, "pipelineLanguage": "zh"},
                        url=CORENLP_SERVER_URL)
    # print(ret.content)
    # print()
    try:
        ret_json = ret.json()["sentences"]
        return ret_json
    except Exception as e:
        print(e)
        return None


PARSING_BUFFER_SIZE = 100


def analyze_file(input_file_name, data_container):
    with open(input_file_name) as file_handle:
        lines = file_handle.readlines()
    sentence_No = 0
    buffer_index = 0
    parsing_sentences = []
    for line in lines:
        sentence = line.strip()
        parsing_sentences.append(sentence)
        buffer_index += 1
        if buffer_index == 100:
            para = "\n".join(parsing_sentences)
            rets = get_parsed_ret(para)
            if not rets:
                continue
            for ret in rets:
                tokens, token_contents = get_tokens(ret)
                data_container.feed_main_data(sentence, sentence_No)
                data_container.feed_tokens_data(token_contents, (sentence_No, 0))
                dependency_tree = build_dependency_tree(ret, tokens)
                # print("#Sentence: %s" % sentence)
                parse_tree(dependency_tree, tokens, (sentence_No, 0), data_container)
                sentence_No += 1
            parsing_sentences = []
            buffer_index = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('--output_path', type=str, default='ret.txt')

    args = parser.parse_args()
    output_path = args.output_path
    input_path = args.input_path
    data_container = Data_container(log_path=output_path, test_mode=1)
    file_num = 0
    if not input_path.endswith('.out') and not input_path.endswith('.txt'):
        exit()
    analyze_file(input_path, data_container)
