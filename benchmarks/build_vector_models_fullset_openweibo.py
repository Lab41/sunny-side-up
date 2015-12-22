from gensim.models import Word2Vec, Doc2Vec
from src.datasets.open_weiboscope import OpenWeibo, OpenWeiboIterator
import cPickle as pickle
import random

# save raw or romanized form
forms = ['pinyin', 'hanzi']
for form in forms:

    # build model vocabulary
    print("Building model for {} text...".format(form))
    sentences = OpenWeiboIterator('/data/openweibo/', form=form)
    model = Word2Vec(size=200, window=5, min_count=1, workers=32)
    model.build_vocab(sentences)
    print("Loaded {} sentences".format(sentences.counter))

    # save model vocab
    model.save_word2vec_format('/data/openweibo/openweibo_fullset_{}_vocabonly{}.bin'.format(form, len(model.vocab)))

    # train model
    sentences = OpenWeiboIterator('/data/openweibo/', form=form)
    model.train(sentences)

    # save model
    model.save_word2vec_format('/data/openweibo/openweibo_fullset_{}_vocab{}.bin'.format(form, len(model.vocab)))
