from gensim.models import Word2Vec, Doc2Vec
from src.datasets.arabic_twitter import ArabicTwitter, ArabicTwitterIterator
import random

# load generator to get raw text of all arabic tweets
sentences = ArabicTwitterIterator('/data/arabic_tweets/arabic_tweets/all.txt')

# build vocabulary
print "building vocabulary..."
model = Word2Vec(size=200, window=5, min_count=10, workers=32)
model.build_vocab(sentences)

# save vocabulary model
print("saving vocabulary-only model ({} terms)...".format(len(model.vocab)))
model.save_word2vec_format('/data/arabic_tweets/arabic_tweets_NLTK_min10vocab_vocabonly{}.bin'.format(len(model.vocab)))

# train model
print "training model..."
sentences = ArabicTwitterIterator('/data/arabic_tweets/arabic_tweets/all.txt')
model.train(sentences)

# save model
print "saving model..."
model.save_word2vec_format('/data/arabic_tweets/arabic_tweets_NLTK_min10vocab_vocab{}.bin'.format(len(model.vocab)), binary=True)
model.save_word2vec_format('/data/arabic_tweets/arabic_tweets_NLTK_min10vocab_vocab{}.nonbinary'.format(len(model.vocab)), binary=False)
