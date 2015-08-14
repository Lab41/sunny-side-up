from textblob import TextBlob
from nltk.corpus import stopwords as stpwrds

stopwords = stpwrds.words('english')

def textblob_ngrams(sentence, n=3, remove_stopwords=False, all_lower_case=False):
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
            List of features of the following form:
                {ngram_1: True, ngram_2: True, ... , ngram_n: True}
    '''

    sentence = TextBlob(sentence)
    features = dict()
    clean_string = ''

    # Changes all word features to lower case if true
    if all_lower_case:
        sentence = sentence.lower()

    # Removes stopwords
    for word in sentence.words:
        # Removes word from features if in nlkt.corpus.stopwords('english')
        if remove_stopwords:
            if word.string in stopwords:
                continue
        clean_string += ''.join([word, ' '])

    for ngram in TextBlob(clean_string):
        features[ngram] = True
    return features
