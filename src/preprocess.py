import re
from urlparse import urlparse
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords as stopCorpus

"""
    Preprocessing methods. Best used with the paradigm:
    import preprocess
    
    preprocess.tweet(stream)
"""

def tweet(text):
    toks = text.split()
    for i in xrange(len(toks)):
        try:
            if urlparse(toks[i]).netloc:
                toks[i] = u"<url>"
        except ValueError:
            pass
    return " ".join(toks)

