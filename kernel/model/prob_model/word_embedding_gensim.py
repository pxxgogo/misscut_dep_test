from gensim.models import KeyedVectors


class Word_vectors:
    def __init__(self, fname):
        self._vectors = KeyedVectors.load_word2vec_format(fname, binary=False)

    def get_similarities(self, key_word, word_list):
        return 1 - self._vectors.distances(key_word, word_list)
