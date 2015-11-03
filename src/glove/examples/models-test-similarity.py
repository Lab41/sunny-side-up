# -*- coding: utf-8 -*-
import argparse

from glove import Glove


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=('Load GloVe model and test similarity scores'))
    parser.add_argument('--model', '-m', action='store',
                        required=True,
                        help='The saved GloVe model object')
    args = parser.parse_args()


    # load the model
    glove = Glove.load_obj(args.model)

    # give me the 5 words most similar to each word in the words list in this
    # corpus and show me how similar the words are in this corpus to each word
    # in the words list in general
    words = ['sky', 'queen', 'car', 'run', 'medical']

    for word in words:

        # get most similar words
        similarities = glove.most_similar(word)

        # print out results
        print("Most similar to {}:".format(word))
        for match, score in similarities:
          print("\t{0:15} {1:.2f}".format(match, score))
