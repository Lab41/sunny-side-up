from gensim.models import Word2Vec, Doc2Vec
from src.datasets.arabic_twitter import ArabicTwitter, ArabicTwitterIterator
import random

# load generator to get raw text of all arabic tweets
sentences = ArabicTwitterIterator('/data/arabic_tweets/arabic_tweets/all.txt')

# build vocabulary
model = Word2Vec(size=200, window=5, min_count=10, workers=32)
model.build_vocab(sentences)
model.save_word2vec_format('/data/arabic_tweets/arabic_tweets_min10vocab_vocabonly{}.bin'.format(len(model.vocab)))

# train model
sentences = ArabicTwitterIterator('/data/arabic_tweets/arabic_tweets/all.txt')
model.train(sentences)

# save model
model.save_word2vec_format('/data/arabic_tweets/arabic_tweets_min10vocab_vocab{}.bin'.format(len(model.vocab)), binary=True)
model.save_word2vec_format('/data/arabic_tweets/arabic_tweets_min10vocab_vocab{}.nonbinary'.format(len(model.vocab)), binary=False)
