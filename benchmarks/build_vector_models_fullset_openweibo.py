from gensim.models import Word2Vec, Doc2Vec
from src.datasets.open_weiboscope import OpenWeibo
from src.datasets.data_utils import tokenize, tokenize_hanzi
import random

# save raw or romanized form
form='hanzi' #pinyin

# load data
data = OpenWeibo('/data/openweibo/').load_data(form=form, keep_retweets=True)

# get input sentences for vector model
if form == 'hanzi':
    sentences = [tokenize_hanzi(text) for text,sentiment in data]
else:
    sentences = [tokenize(text) for text,sentiment in data]
print("loaded {} sentences".format(len(sentences)))

# build and train model
model = Word2Vec(size=200, window=5, min_count=1, workers=32)
model.build_vocab(sentences)
model.train(sentences)

# save model
model.save_word2vec_format('/data/openweibo/openweibo_fullset_{}_vocab{}.bin'.format(form, len(model.vocab)))
