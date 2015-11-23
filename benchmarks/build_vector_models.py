from src.datasets.data_utils import WordVectorBuilder
from src.datasets.open_weiboscope import OpenWeibo
from src.datasets.amazon_reviews import AmazonReviews
from src.datasets.sentiment140 import Sentiment140

builder = WordVectorBuilder(Sentiment140, '/data/sentiment140.csv')
builder.word2vec_save('/data/sentiment140.bin'.format(min_samples), min_samples=min_samples)
