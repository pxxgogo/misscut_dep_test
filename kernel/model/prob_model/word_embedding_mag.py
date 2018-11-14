import io
import numpy as np
from pymagnitude import *


SCORE_THRESHOLD = 0.5
CACHE_SIZE = 1000

class Word_vectors:
    def __init__(self, fname, words_num=-1):
        self._vectors = Magnitude(fname, lazy_loading=-1,
                            blocking=True)

    def get_similarities(self, key_word, word_list):
        return self._vectors.similarity(key_word, word_list)
