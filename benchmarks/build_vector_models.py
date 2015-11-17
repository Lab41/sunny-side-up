from src.datasets.data_utils import WordVectorBuilder
from src.datasets.open_weiboscope import OpenWeibo

builder = WordVectorBuilder(OpenWeibo, '/data/openweibo/')
builder.word2vec_save('/data/weibo.bin', min_samples=800000)
