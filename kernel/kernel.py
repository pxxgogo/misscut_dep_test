import argparse
import os
import re
import requests
from .ngram import Ngram

CORENLP_SERVER_URL = 'http://166.111.139.15:9000'

HEADER_PATTERN = r'Sentence #(\d+) \((\d+) tokens\):'
HEADER_COMPILER = re.compile(HEADER_PATTERN)

TOKEN_PATTERN = r'\[Text=(\S+) CharacterOffsetBegin=(\d+) CharacterOffsetEnd=(\d+) PartOfSpeech=(\w+)\]'
TOKEN_COMPILER = re.compile(TOKEN_PATTERN)

DEPENDENCY_HEADER_PREFIX = 'Dependency Parse'

DEPENDENCY_ITEM_PATTERN = r'(\w+)\((\S+)-(\d+), (\S+)-(\d+)\)'
DEPENDENCY_ITEM_COMPILER = re.compile(DEPENDENCY_ITEM_PATTERN)
SENTENCE_MAX_LENGTH = 50

ngram_model = None


def get_scores(type, word_0, label_0, word_1, label_1, word_2):
    scores = ngram_model.score(word_0, label_0, word_1, label_1, word_2, type)
    sum_score = 0
    for score_info in scores:
        sum_score += score_info[0]
    return scores, sum_score


def search_beginning(lines, line_No, lines_num):
    while line_No < lines_num:
        header = lines[line_No].strip()
        header_ret_list = HEADER_COMPILER.findall(header)
        if len(header_ret_list) != 0:
            return line_No, header_ret_list
        line_No += 1
    return -1, None


def operate_sub_tree(type, word_0, label_0, word_1, label_1, word_2):
    scores, sum_score = get_scores(type, word_0, label_0, word_1, label_1, word_2)
    if type == 0:
        type_word = 'D'
    else:
        type_word = 'B'
    if len(scores) == 6:
        ret_sentence = "%s: %s [%.2f, %d] %s [%.2f, %d] %s [%.2f, %d] %s [%.2f, %d] %s [%.2f, %d] # %.2f \n" % (
            type_word, word_0, scores[0][0], scores[0][1], label_0, scores[1][0], scores[1][1],
            word_1, scores[2][0], scores[2][1], label_1, scores[3][0], scores[3][1], word_2,
            scores[4][0], scores[4][1], sum_score)
    else:
        ret_sentence = "%s: %s [%.2f, %d] %s [%.2f, %d] %s [%.2f, %d] # %.2f \n" % (
            type_word, word_0, scores[0][0], scores[0][1], word_1, scores[1][0], scores[1][1],
            word_2, scores[2][0], scores[2][1], sum_score)

    return ret_sentence, sum_score


def search_pattern_1(node_No, tokens, tree, output_file_handle, scores):
    node_info = tree[node_No]
    for child_info in node_info:
        for grandchild_info in tree[child_info['child']]:
            output_sentence, score = operate_sub_tree(0, tokens[node_No]['text'], child_info['relation'],
                                               tokens[child_info['child']]['text'], grandchild_info['relation'],
                                               tokens[grandchild_info['child']]['text'])
            output_file_handle.write(output_sentence)
            scores.append(score)


def search_pattern_2(node_No, tokens, tree, output_file_handle, scores):
    node_info = tree[node_No]
    child_num = len(node_info)
    for child_1_No in range(child_num):
        for child_2_No in range(child_1_No + 1, child_num):
            child_1_info = node_info[child_1_No]
            child_2_info = node_info[child_2_No]
            output_sentence, score = operate_sub_tree(1, tokens[node_No]['text'], child_1_info['relation'],
                                               tokens[child_1_info['child']]['text'], child_2_info['relation'],
                                               tokens[child_2_info['child']]['text'])
            output_file_handle.write(output_sentence)
            scores.append(score)


def parse_node(node_No, tokens, tree, output_file_handle, scores):
    search_pattern_1(node_No, tokens, tree, output_file_handle, scores)
    search_pattern_2(node_No, tokens, tree, output_file_handle, scores)
    for child_info in tree[node_No]:
        parse_node(child_info['child'], tokens, tree, output_file_handle, scores)


def parse_tree(tree, tokens, output_file_handle):
    scores = []
    parse_node(0, tokens, tree, output_file_handle, scores)
    return scores


def get_tokens(ret_json):
    tokens = []
    tokens.append({'text': '{ROOT}', 'index': (-1, -1), 'POS': 'ROOT'})
    tokens_json = ret_json["tokens"]
    for token_info in tokens_json:
        tokens.append({'text': token_info["originalText"],
                       'index': (token_info["characterOffsetBegin"], token_info["characterOffsetEnd"]),
                       'POS': token_info["pos"]})
    return tokens


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
        relation = relation[critical_position+1:]
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


def get_parsed_ret(line):
    properties = '{"annotators":"tokenize,ssplit,pos,depparse","outputFormat":"json"}'
    # print(line)
    data = line
    ret = requests.post(data=data.encode("utf-8"), params={"properties": properties, "pipelineLanguage": "zh"},
                        url=CORENLP_SERVER_URL)
    # print(ret.content)
    # print()
    return ret.json()["sentences"][0]


def analyze_file(input_file_name, output_file_name):
    with open(input_file_name) as file_handle:
        lines = file_handle.readlines()
    output_file_handle = open(output_file_name, 'w')
    for line in lines:
        sentence = line.strip()
        ret = get_parsed_ret(sentence)
        tokens = get_tokens(ret)
        dependency_tree = build_dependency_tree(ret, tokens)
        # print("#Sentence: %s" % sentence)
        output_file_handle.write("#Sentence: %s\n" % sentence)
        parse_tree(dependency_tree, tokens, output_file_handle)
    output_file_handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('--output_path', type=str, default='ret.txt')
    parser.add_argument('--model_flag', type=int, default=0)

    args = parser.parse_args()
    output_path = args.output_path
    input_path = args.input_path
    model_flag = args.model_flag
    ngram_model = Ngram(model_flag)
    file_num = 0
    if not input_path.endswith('.out') and not input_path.endswith('.txt'):
        exit()
    analyze_file(input_path, output_path)
