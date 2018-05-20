import io
import numpy as np
from scipy.spatial import cKDTree

SCORE_THRESHOLD = 0.5

class Word_vectors:
    def __init__(self, fname):
        self.word2vector = {}
        self.words = []
        self.vectors = []
        self.kdtree = None
        self._load_vectors(fname)
        self._build_kdtree()

    def _load_vectors(self, fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        for line in fin:
            tokens = line.rstrip().split(' ')
            raw_vector = list(map(float, tokens[1:]))
            vector = np.array(raw_vector, np.float16)
            self.word2vector[tokens[0]] = vector
            self.vectors.append(vector)
            self.words.append(tokens[0])
        self.vectors = np.array(self.vectors, np.float16)

    def _build_kdtree(self):
        self.kdtree = cKDTree(self.vectors)

    def get_word_vector(self, word):
        if not word in self.word2vector:
            return np.zeros([300])
        vector = self.word2vector[word]
        vector /= np.sqrt(vector.dot(vector))
        return vector

    def get_closed_words(self, word_name, k=30):
        key_word_vector = self.get_word_vector(word_name)
        query = self.kdtree.query(key_word_vector, k)
        indexes = query[1]
        rets = []
        for index in indexes:
            word = self.words[index]
            vector = self.get_word_vector(word)
            score = vector.dot(key_word_vector)
            if score > SCORE_THRESHOLD:
                rets.append((word, score))
            else:
                break
        return rets

