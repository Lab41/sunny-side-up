import re
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords as stopCorpus

re_url = re.compile(ur"https?:\/\/\S+\b|www\.(\w+\.)+\S*", re.MULTILINE | re.DOTALL)

"""
    Preprocessing methods. Best used with the paradigm:
    import preprocess
    
    preprocess.tweet(stream)
"""

def tweet(text):
    return re_url.sub(text, u"<url>")

