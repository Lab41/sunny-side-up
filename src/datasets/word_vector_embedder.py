import os
import numpy as np
from glove.glove import Glove
from gensim.models import Doc2Vec
from model_downloader import ModelDownloader

class WordVectorEmbedder:
    '''
        generic class to embed words into word vectors
    '''


    def __init__(self, model_type, model_fullpath=None, model_group=None, model_subset=None):
        '''
            initialize a model from a saved object file
        '''
        if model_type == 'word2vec':

            # default model
            if model_fullpath is None:
                model_dir       = '/data'
                model_group     = 'google-news'
                model_subset    = 'GoogleNews-vectors-negative300.bin'
                model_args      = { 'binary': True }

            # setup importer and converter
            self.model_import_method = Doc2Vec.load_word2vec_format
            self.word_vector = self.word_vector_word2vec

        elif model_type == 'glove':

            # default model
            if model_fullpath is None:
                model_dir       = '/data'
                model_group     = 'twitter-2b'
                model_subset    = 'glove.twitter.27B.25d'
                model_args      = {}

            # setup importer and converter
            self.model_import_method = Glove.load_obj
            self.word_vector = self.word_vector_glove

        else:
            raise NameError("Error! You must specify a model type from: <word2vec|glove>")

        # download and save the model (ModelDownloader will skip if exists)
        downloader = ModelDownloader(model_type)
        downloader.download_and_save(outdir=model_dir, datafile=model_subset, dataset=model_group)

        # locate the model
        model_fullpath = downloader.download_fullpath(model_dir, model_subset)

        # load the model
        print("Loading model from {}...".format(model_fullpath))
        self.model = self.model_import_method(model_fullpath, **model_args)


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


    def embed_words_into_vectors(self, text):
        '''
            embed text into glove's vector space
        '''

        # store vectors as list
        vectors = []

        # process tokens
        for word in text.split():
            try:

                # add vector
                vectors.append(self.word_vector(word))

            # ignore words not in dictionary
            except KeyError as e:
                pass

        # return list of embedded words
        return np.array(vectors)
