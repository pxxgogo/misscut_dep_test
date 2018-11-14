import pymagnitude


class Word_vectors:
    def __init__(self, fname):
        self._vectors = pymagnitude.Magnitude(fname, lazy_loading=-1, blocking=True)

    def get_similarities(self, key_word, word_list):
        return self._vectors.similarity(key_word, word_list)
