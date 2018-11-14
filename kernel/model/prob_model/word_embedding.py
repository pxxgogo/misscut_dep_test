import io
import numpy as np

SCORE_THRESHOLD = 0.5
CACHE_SIZE = 1000

class Word_vectors:
    def __init__(self, fname, words_num=-1):
        self.word2vector = {}
        self.words = []
        self.vectors = []
        self.kdtree = None
        self._load_vectors(fname, words_num)
        # self._build_kdtree()
        # self.cache_similar_words_info = {}
        # self.cache_keys = []
        # self.cache_key_index = 0

    def _load_vectors(self, fname, words_num):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        word_No = 0
        for line in fin:
            if words_num != -1 and word_No > words_num:
                break
            tokens = line.rstrip().split(' ')
            raw_vector = list(map(float, tokens[1:]))
            vector = np.array(raw_vector, np.float16)
            vector /= np.sqrt(vector.dot(vector))
            self.word2vector[tokens[0]] = vector
            self.vectors.append(vector)
            self.words.append(tokens[0])
            word_No += 1
        self.vectors = np.array(self.vectors, np.float16)

    # def _build_kdtree(self):
    #     self.kdtree = cKDTree(self.vectors)

    def get_word_vector(self, word):
        if not word in self.word2vector:
            return False, np.zeros(300)
        vector = self.word2vector[word]
        return True, vector

    # def get_closed_words(self, word_name, k=30):
    #     if word_name in self.cache_similar_words_info:
    #         return self.cache_similar_words_info[word_name]
    #     key_word_vector = self.get_word_vector(word_name)
    #     query = self.kdtree.query(key_word_vector, k)
    #     indexes = query[1]
    #     rets = []
    #     for index in indexes:
    #         word = self.words[index]
    #         if word == word_name:
    #             continue
    #         vector = self.get_word_vector(word)
    #         score = vector.dot(key_word_vector)
    #         if score > SCORE_THRESHOLD:
    #             rets.append((word, score))
    #         else:
    #             break
    #     if self.cache_key_index >= len(self.cache_keys):
    #         self.cache_keys.append(word_name)
    #         self.cache_similar_words_info[word_name] = rets
    #     else:
    #         self.cache_similar_words_info.pop(self.cache_keys[self.cache_key_index])
    #         self.cache_similar_words_info[word_name] = rets
    #         self.cache_keys[self.cache_key_index] = word_name
    #     self.cache_key_index += 1
    #     self.cache_key_index = self.cache_key_index % CACHE_SIZE
    #     return rets