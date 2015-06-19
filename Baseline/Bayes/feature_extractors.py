''' The file is a collection of functions that take in a sentence string
    as input, and output a set of features related to that sentence

    Output of each function is of the following format:
        {feature_1, feature_2, ... , feature_n}
'''

from nltk import word_tokenize
from nltk.corpus import stopwords as stpwrds

stopwords = stpwrds.words('english')


def word_feats(sentence, tokenizer=word_tokenize, remove_stopwords=False,
               stemmer=None, all_lower_case=False):
    ''' Takes in a sentence returns the words and/or punctuation
        in that sentence as the features (depending on chosen tokenizer)

        @Arguments:
            sentence -- Chosen sentence to tokenize, type(sentence) = String

            tokenizer (optional) -- Function of type nlkt.tokenize to be used
                for breaking apart the sentence string. Standard tokenizer
                splits on whitespace and removes punctuation

            remove_stopwords (optional) -- if true, all stopwords in sentence
                will not be included as features. Currently only for English
                text. Value is initially false

            stemmer (optional) -- Function of type nltk.stem to be used for
                stemming word features.

        @Return:
            List of features of the follwing form:
                {word_1: True, word_2: True, ... , word_n: True}
    '''

    features = dict()
    for word in tokenizer(sentence):
        # Removes word from features if in nlkt.corpus.stopwords('english')
        if remove_stopwords:
            if word.lower() in stopwords:
                continue

        # Changes all word features to lower case if ture
        if all_lower_case:
            word = word.lower()
        # Stems all word features with chosen stemmer function if not None
        if stemmer:
            word = stemmer(word)
        features[word] = True
    return features
