import io
import numpy as np

SCORE_THRESHOLD = 0.5
CACHE_SIZE = 1000


class Word_vectors:
    def __init__(self, fname, words_num=-1, gpu_flag=0):
        self._word2index = {}
        self._words = []
        self._vectors = None
        # self.kdtree = None
        self._dimension = 0
        if gpu_flag == 1:
            import cupy as cp
            self._op = cp
        else:
            self._op = np
        self._load_vectors(fname, words_num)

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
        self._vectors = self._op.array(vectors, np.float16)
        del vectors

    def is_in_vocab(self, word):
        return word in self._word2index

    def get_similarities(self, key_word, word_list):
        key_vector = self._vectors[self._word2index[key_word]]
        other_indices = [self._word2index[word] for word in word_list]
        other_vectors = self._vectors[other_indices]
        return self._op.dot(other_vectors, key_vector)
