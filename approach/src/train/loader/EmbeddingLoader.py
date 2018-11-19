import logging
import numpy as np
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)

embed_size_word2vec = 256
min_word_frequency_word2vec = 1


class EmbeddingLoader:
    def __init__(self, namespace, container):
        self._ns = namespace
        self._c = container

    def train(self):
        logger.info("EmbeddingLoader has no training step")
        return self._c

    def load(self):
        all_data = np.array(self._c['data'])

        wordvec_model = Word2Vec(sentences=all_data,
                                 min_count=min_word_frequency_word2vec,
                                 size=embed_size_word2vec)

        self._c['vocabulary'] = wordvec_model.wv.vocab

        # convert the wv word vectors into a numpy matrix that is suitable for insertion
        # into our TensorFlow and Keras models
        embedding_matrix = np.zeros((len(wordvec_model.wv.vocab), embed_size_word2vec))
        for i in range(len(wordvec_model.wv.vocab)):
            embedding_vector = wordvec_model.wv[wordvec_model.wv.index2word[i]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        self._c['embedding_matrix'] = embedding_matrix

        return self._c


def load_word2vec(documents):
    wordvec_model = Word2Vec(sentences=documents,
                             min_count=min_word_frequency_word2vec,
                             size=embed_size_word2vec)

    # convert the wv word vectors into a numpy matrix that is suitable for insertion
    # into our TensorFlow and Keras models
    print('vocab: ')
    print(len(wordvec_model.wv.vocab))
    embedding_matrix = np.zeros((len(wordvec_model.wv.vocab), embed_size_word2vec))
    for i in range(len(wordvec_model.wv.vocab)):
        embedding_vector = wordvec_model.wv[wordvec_model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return wordvec_model, embedding_matrix
