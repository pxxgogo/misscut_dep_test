import io
import numpy as np

SCORE_THRESHOLD = 0.5
CACHE_SIZE = 1000

class Word_vectors:
    def __init__(self, fname, words_num=-1):
        self._word2index = {}
        self._words = []
        self._vectors = None
        # self.kdtree = None
        self._dimension = 0
        self._load_vectors(fname, words_num)
        # self._build_kdtree()
        # self.cache_similar_words_info = {}
        # self.cache_keys = []
        # self.cache_key_index = 0

    def _load_vectors(self, fname, words_num):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        self._dimension = d
        word_No = 0
        vectors = []
        for line in fin:
            if words_num != -1 and word_No > words_num:
                break
            tokens = line.rstrip().split(' ')
            raw_vector = list(map(float, tokens[1:]))
            vector = np.array(raw_vector, np.float16)
            vector /= np.sqrt(vector.dot(vector))
            self._word2index[tokens[0]] = word_No
            vectors.append(vector)
            self._words.append(tokens[0])
            word_No += 1
        self._vectors = np.array(vectors, np.float16)
        del vectors

    # def _build_kdtree(self):
    #     self.kdtree = cKDTree(self.vectors)

    def is_in_vocab(self, word):
        return word in self._word2index


    def get_similarities(self, key_word, word_list):
        key_vector = self._vectors[self._word2index[key_word]]
        other_indices = [self._word2index[word] for word in word_list]
        other_vectors = self._vectors[other_indices]
        return np.dot(other_vectors, key_vector)

    # def get_closed_words(self, word_name, k=30):
    #     if word_name in self.cache_similar_words_info:
    #         return self.cache_similar_words_info[word_name]
    #     key_word_vector = self.get_word_vector(word_name)
    #     query = self.kdtree.query(key_word_vector, k)
    #     indexes = query[1]
    #     rets = []
    #     for index in indexes:
    #         word = self._words[index]
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