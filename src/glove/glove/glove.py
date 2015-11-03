# GloVe model from the NLP lab at Stanford:
# http://nlp.stanford.edu/projects/glove/.
import array
import collections
import io
try:
    # Python 2 compat
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import scipy.sparse as sp
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#from .glove_cython import fit_vectors, transform_paragraph
from glove_cython import fit_vectors, transform_paragraph


class Glove(object):
    """
    Class for estimating GloVe word embeddings using the
    corpus coocurrence matrix.
    """

    def __init__(self, no_components=30, learning_rate=0.05,
                 alpha=0.75, max_count=100, max_loss=10.0):
        """
        Parameters:
        - int no_components: number of latent dimensions
        - float learning_rate: learning rate for SGD estimation.
        - float alpha, float max_count: parameters for the
          weighting function (see the paper).
        - float max_loss: the maximum absolute value of calculated
                          gradient for any single co-occurrence pair.
                          Only try setting to a lower value if you
                          are experiencing problems with numerical
                          stability.
        """

        self.no_components = no_components
        self.learning_rate = float(learning_rate)
        self.alpha = float(alpha)
        self.max_count = float(max_count)
        self.max_loss = max_loss

        self.word_vectors = None
        self.word_biases = None

        self.vectors_sum_gradients = None
        self.biases_sum_gradients = None

        self.dictionary = None
        self.inverse_dictionary = None

        self.pca = None

    def fit(self, matrix, epochs=5, no_threads=2, verbose=False):
        """
        Estimate the word embeddings.

        Parameters:
        - scipy.sparse.coo_matrix matrix: coocurrence matrix
        - int epochs: number of training epochs
        - int no_threads: number of training threads
        - bool verbose: print progress messages if True
        """

        shape = matrix.shape

        if (len(shape) != 2 or
            shape[0] != shape[1]):
            raise Exception('Coocurrence matrix must be square')

        if not sp.isspmatrix_coo(matrix):
            raise Exception('Coocurrence matrix must be in the COO format')

        self.word_vectors = ((np.random.rand(shape[0],
                                             self.no_components) - 0.5)
                                             / self.no_components)
        self.word_biases = np.zeros(shape[0],
                                    dtype=np.float64)

        self.vectors_sum_gradients = np.ones_like(self.word_vectors)
        self.biases_sum_gradients = np.ones_like(self.word_biases)

        shuffle_indices = np.arange(matrix.nnz, dtype=np.int32)

        if verbose:
            print('Performing %s training epochs '
                  'with %s threads' % (epochs, no_threads))

            # initialize lists that will hold the learning rates
            vectors_gradients = list()
            biases_gradients = list()

        for epoch in range(epochs):

            if verbose:
                starttime = dt.datetime.now()
                print('Epoch %s' % epoch)

            # Shuffle the coocurrence matrix
            np.random.shuffle(shuffle_indices)

            fit_vectors(self.word_vectors,
                        self.vectors_sum_gradients,
                        self.word_biases,
                        self.biases_sum_gradients,
                        matrix.row,
                        matrix.col,
                        matrix.data,
                        shuffle_indices,
                        self.learning_rate,
                        self.max_count,
                        self.alpha,
                        self.max_loss,
                        int(no_threads))

            if not np.isfinite(self.word_vectors).all():
                raise Exception('Non-finite values in word vectors. '
                                'Try reducing the learning rate or the '
                                'max_loss parameter.')

            if verbose:
                vectors_gradients.append(np.mean([self.learning_rate/np.sqrt(a) for a in self.vectors_sum_gradients]))
                biases_gradients.append(np.mean(self.learning_rate/np.sqrt(self.biases_sum_gradients)))

                endtime = dt.datetime.now()
                print('    Epoch %s took %s minutes' % (epoch, (endtime-starttime).total_seconds() / 60))

        if verbose:
            # show the learning rates
            plt.plot(vectors_gradients, 'k--', biases_gradients, 'k:')
            plt.legend(('word vectors', 'word biases'))
            plt.xlabel('Epoch')
            plt.ylabel('Mean learning rate')
            plt.title('Change in mean learning rates across epochs')
            plt.show()

    def perform_PCA(self, num_components=None):
        """
        Performs PCA on the word vectors in the model
        Uses the number of components passed in if any
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit trying to perform PCA')

        if num_components:
            self.pca = PCA(n_components=num_components)
        else:
            self.pca = PCA()

        self.pca.fit(self.word_vectors)

    def transform_paragraph(self, paragraph, epochs=50, ignore_missing=False, use_pca=False):
        """
        Transform an iterable of tokens into its vector representation
        (a paragraph vector).

        Experimental. This will return something close to a tf-idf
        weighted average of constituent token vectors by fitting
        rare words (with low word bias values) more closely. If use_pca is True,
        the token vectors will be transformed using PCA before the weighted
        average is calculated.

        Added the PCA thing because of this paper: http://courses.cs.tau.ac.il/~wolf/papers/qagg.pdf
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit to transform paragraphs')

        if self.dictionary is None:
            raise Exception('Dictionary must be provided to '
                            'transform paragraphs')

        cooccurrence = collections.defaultdict(lambda: 0.0)

        for token in paragraph:
            try:
                cooccurrence[self.dictionary[token]] += self.max_count / 10.0
            except KeyError:
                if not ignore_missing:
                    raise

        word_ids = np.array(cooccurrence.keys(), dtype=np.int32)
        values = np.array(cooccurrence.values(), dtype=np.float64)
        shuffle_indices = np.arange(len(word_ids), dtype=np.int32)

        # Initialize the vector to mean of constituent word vectors
        # if a PCA model should be used, use it first to transform the vectors

        if use_pca:
            if self.pca is None:
                self.perform_PCA()

            paragraph_vector = np.mean(self.pca.transform(self.word_vectors[word_ids]), axis=0)
        else:
            paragraph_vector = np.mean(self.word_vectors[word_ids], axis=0)

        sum_gradients = np.ones_like(paragraph_vector)

        # Shuffle the coocurrence matrix
        np.random.shuffle(shuffle_indices)
        transform_paragraph(self.word_vectors,
                            self.word_biases,
                            paragraph_vector,
                            sum_gradients,
                            word_ids,
                            values,
                            shuffle_indices,
                            self.learning_rate,
                            self.max_count,
                            self.alpha,
                            epochs)

        return paragraph_vector

    def add_dictionary(self, dictionary):
        """
        Supply a word-id dictionary to allow similarity queries.
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit before adding a dictionary')

        if len(dictionary) > self.word_vectors.shape[0]:
            raise Exception('Dictionary length must be smaller '
                            'or equal to the number of word vectors')

        self.dictionary = dictionary
        if hasattr(self.dictionary, 'iteritems'):
            # Python 2 compat
            items_iterator = self.dictionary.iteritems()
        else:
            items_iterator = self.dictionary.items()

        self.inverse_dictionary = {v: k for k, v in items_iterator}

    def save(self, filename):
        """
        Serialize model to filename.
        """

        with open(filename, 'wb') as savefile:
            pickle.dump(self.__dict__,
                        savefile,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def save_obj(self, filename):
        """
        save glove object to file
        """

        with open(filename, 'wb') as savefile:
            pickle.dump(self, savefile, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_obj(cls, filename):
        """
        load glove object from file
        """

        instance = Glove()

        with open(filename, 'rb') as savefile:
            instance = pickle.load(savefile)

        return instance


    @classmethod
    def load(cls, filename):
        """
        Load model from filename.
        """

        instance = Glove()

        with open(filename, 'rb') as savefile:
            instance.__dict__ = pickle.load(savefile)

        return instance

    @classmethod
    def load_stanford(cls, filename):
        """
        Load model from the output files generated by
        the C code from http://nlp.stanford.edu/projects/glove/.

        The entries of the word dictionary will be of type
        unicode in Python 2 and str in Python 3.
        """

        dct = {}
        vectors = array.array('d')

        # Read in the data.
        with io.open(filename, 'r', encoding='utf-8') as savefile:
            for i, line in enumerate(savefile):
                tokens = line.split(' ')

                word = tokens[0]
                entries = tokens[1:]

                dct[word] = i
                vectors.extend(float(x) for x in entries)

        # Infer word vectors dimensions.
        no_components = len(entries)
        no_vectors = len(dct)

        # Set up the model instance.
        instance = Glove()
        instance.no_components = no_components
        instance.word_vectors = (np.array(vectors)
                                 .reshape(no_vectors,
                                          no_components))
        instance.word_biases = np.zeros(no_vectors)
        instance.add_dictionary(dct)

        return instance

    def _similarity_query(self, word_vec, number, similar=True, show_hist=False):

        dst = (np.dot(self.word_vectors, word_vec)
               / np.linalg.norm(self.word_vectors, axis=1)
               / np.linalg.norm(word_vec))
        word_ids = np.argsort(-dst)

        # build histogram
        n, bins, patches = plt.hist(dst, range=(-1, 1), weights=np.ones_like(dst)/float(len(dst)), facecolor='green', alpha=0.5)
        plt.xlabel('Similarity')
        plt.ylabel('Probability')
        plt.title('Histogram of word similarities for `'+self.inverse_dictionary[word_ids[0]]+'`')

        # show or save
        if show_hist:
            plt.show()
        else:
            plt.savefig('./similarity_vs_probability.png')

        if similar:
            return [(self.inverse_dictionary[x], dst[x]) for x in word_ids[:number]
                if x in self.inverse_dictionary]
        else:
            return [(self.inverse_dictionary[x], dst[x]) for x in word_ids[-number:]
                if x in self.inverse_dictionary]

    def most_similar(self, word, number=5, show_hist=False):
        """
        Run a similarity query, retrieving number
        most similar words.
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit before querying')

        if self.dictionary is None:
            raise Exception('No word dictionary supplied')

        try:
            word_idx = self.dictionary[word]
        except KeyError:
            raise Exception('Word not in dictionary')

        #do this weirdness to remove the word itself from the vector and to get the next n words
        return self._similarity_query(self.word_vectors[word_idx], number+1, show_hist=show_hist)[1:]

    def least_similar(self, word, number=5, show_hist=False):
        """
        Run a similarity query, retrieving number
        least similar words.
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit before querying')

        if self.dictionary is None:
            raise Exception('No word dictionary supplied')

        try:
            word_idx = self.dictionary[word]
        except KeyError:
            raise Exception('Word not in dictionary')

        #return self._similarity_query(self.word_vectors[word_idx], number)[1:]

        #do this weirdness to remove the word itself from the vector and to get the next n words
        return self._similarity_query(self.word_vectors[word_idx], number, False, show_hist)

    def most_similar_paragraph(self, paragraph, number=5, **kwargs):
        """
        Return words most similar to a given paragraph (iterable of tokens).
        """

        paragraph_vector = self.transform_paragraph(paragraph, **kwargs)

        return self._similarity_query(paragraph_vector, number)


    def get_similarity_score(self, word1, word2):
        """
        Returns the similarity score of two words
        Both words must be in the dictionary
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit before querying')

        if self.dictionary is None:
            raise Exception('No word dictionary supplied')

        try:
            word_idx1 = self.dictionary[word1]
            word_idx2 = self.dictionary[word2]
        except KeyError:
            raise Exception('One of your words was not in dictionary')


        dst = (np.dot(self.word_vectors[word_idx1], self.word_vectors[word_idx2])
        / np.linalg.norm(self.word_vectors[word_idx1])
        / np.linalg.norm(self.word_vectors[word_idx2]))

        return dst
