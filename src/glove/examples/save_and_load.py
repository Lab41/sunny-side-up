# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:28:10 2015
@author: brittanyl
"""

from __future__ import print_function

from glove import Glove
from glove import Corpus


# turns corpus into iterable
# uses a very simple tokenizer and removes non-ascii characters
def read_corpus(filename):

    delchars = [chr(c) for c in range(256)]
    delchars = [x for x in delchars if not x.isalnum()]
    delchars.remove(' ')
    delchars = ''.join(delchars)

    with open(filename, 'r') as datafile:
        for line in datafile:
            # list of tokenized words
            yield line.lower().translate(None, delchars).split(' ')


if __name__ == '__main__':

    # initialize glove object
    glove = Glove(no_components=100, learning_rate=0.05)

    # read in the data to train on; this file is shakespeare text
    corpus_model = Corpus()
    corpus_model.fit(read_corpus("data/input.txt"), window=10)

    # fit the model using the given parameters
    glove.fit(corpus_model.matrix, epochs=10, no_threads=1, verbose=True)

    # add a dictionary just to make it easier for similarity queries
    glove.add_dictionary(corpus_model.dictionary)

    # save glove object to file
    glove.save_obj('glove.model.obj')

    # give me the 5 words most similar to each word in the words list in this 
    # corpus and show me how similar the words are in this corpus to each word
    # in the words list in general
    words = ['sky', 'queen', 'car']

    for word in words:
        glove.most_similar(word, show_hist=True)
