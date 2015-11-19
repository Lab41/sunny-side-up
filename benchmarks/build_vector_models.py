from src.datasets.data_utils import WordVectorBuilder
from src.datasets.open_weiboscope import OpenWeibo
from src.datasets.amazon_reviews import AmazonReviews

min_samples = 100000
#builder = WordVectorBuilder(OpenWeibo, '/data/openweibo/')
#builder.word2vec_save('/data/openweibo/openweibo_{}.bin'.format(min_samples), min_samples=min_samples)

builder = WordVectorBuilder(AmazonReviews, '/data/amazon/amazonreviews.gz')
builder.word2vec_save('/data/amazon/amazon_{}.bin'.format(min_samples), min_samples=min_samples)
