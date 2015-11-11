import os
import numpy as np
from glove.glove import Glove
from gensim.models import Doc2Vec
from model_downloader import ModelDownloader
from data_utils import TextTooShortException

class WordVectorEmbedder:
    '''
        generic class to embed words into word vectors
    '''


    def __init__(self, model_type, model_fullpath=None, model_group=None, model_subset=None):
        '''
            initialize a model from a saved object file
        '''
        self.model_type = model_type
        if self.model_type == 'word2vec':

            # default model
            if model_fullpath is None:
                model_dir       = '/data'
                model_group     = 'google-news'
                model_subset    = 'GoogleNews-vectors-negative300.bin'
                model_args      = { 'binary': True }

            # setup importer and converter
            self.model_import_method = Doc2Vec.load_word2vec_format
            self.word_vector = self.word_vector_word2vec

        elif self.model_type == 'glove':

            # default model
            if model_fullpath is None:
                model_dir       = '/data'
                model_group     = 'twitter-2b'
                model_subset    = 'glove.twitter.27B.200d'
                model_args      = {}

            # setup importer and converter
            self.model_import_method = Glove.load_obj
            self.word_vector = self.word_vector_glove

        else:
            raise NameError("Error! You must specify a model type from: <word2vec|glove>")

        # save subset for documentation
        self.model_subset = model_subset

        # download and save the model (ModelDownloader will skip if exists)
        downloader = ModelDownloader(self.model_type)
        downloader.download_and_save(outdir=model_dir, datafile=model_subset, dataset=model_group)

        # locate the model
        model_fullpath = downloader.download_fullpath(model_dir, model_subset)

        # load the model
        print("Loading model from {}...".format(model_fullpath))
        self.model = self.model_import_method(model_fullpath, **model_args)

        # setup the word lookup
        if self.model_type == 'word2vec':
            self.word_set = set(self.model.index2word)
        else:
            self.word_set = set(self.model.dictionary)


    def num_features(self):
        if self.model_type == 'word2vec':
            return self.model.vector_size
        else:
            return self.model.no_components


    def word_vector_glove(self, word):
        '''
            get glove vector for given word
        '''
        word_idx = self.model.dictionary[word]
        return self.model.word_vectors[word_idx]


    def word_vector_word2vec(self, word):
        '''
            get glove vector for given word
        '''
        return self.model[word]


    def embed_words_into_vectors(self, words, num_features=None):
        '''
            embed words into model's vector space
        '''

        # store vectors as list
        vectors = []

        # process tokens
        for word in words:
            try:

                # add vector
                vectors.append(self.word_vector(word))

            # ignore words not in dictionary
            except KeyError as e:
                pass


        # build fixed-length set if necessary
        if num_features:

            # truncate if longer
            if (len(vectors) >= num_features):
                vectors = vectors[:num_features]

            # pad if necessary by appending right-sized 0 vectors
            else:
                padding_length = num_features - len(vectors)
                for i in xrange(padding_length):
                    vectors.append(np.zeros(num_features*self.num_features()))

        # return ndarray of embedded words
        return np.array(vectors)


    def embed_words_into_vectors_concatenated(self, words, num_features=None):
        vectors = self.embed_words_into_vectors(words, num_features)
        return vectors.flatten()


    def embed_words_into_vectors_averaged(self, words):
        '''
            embed words into model's averaged vector space
        '''
        # Function to average all of the word vectors in a given
        # paragraph

        # choose model
        if self.model_type == 'glove':
            vector = self.model.transform_paragraph(words, ignore_missing=True, epochs=0)
            return np.nan_to_num(vector)
        else:

            # process valid words
            valid_words = [word for word in words if word in self.word_set]
            if len(valid_words):

                # get vectors for valid words
                vectors = self.word_vector(valid_words)

                # find the average/paragraph vector
                return np.mean(vectors, axis=0)

            else:
                raise TextTooShortException()
