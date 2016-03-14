# sunny-side-up

Lab41's foray into Sentiment Analysis with Deep Learning.  In addition to checking out the source code, visit the [Wiki](https://github.com/Lab41/sunny-side-up/wiki) for Learning Resources and possible Conferences to attend.

> Try them, try them, and you may! Try them and you may, I say.

## Table of Contents
- [Blog Overviews](#blog-overviews)
- [Docker Environments](#docker-environments)
- [Binary Classification with Word Vectors](#binary-classification-with-word-vectors)
  - Word Vector Models
  - Training and Testing Data
- [Binary Classification via Deep Learning](#binary-classification-via-deep-learning)
  - CNN
  - LSTM


### Blog Overviews
- [Can Word Vectors Help Predict Whether Your Chinese Tweet Gets Censored?](http://www.lab41.org/will-your-chinese-tweet-get-censored/) - March 2016
- [One More Reason Not To Be Scared of Deep Learning](http://www.lab41.org/one-more-reason-not-to-be-scared-of-deep-learning/) - March 2016
- [Some Tips for Debugging in Deep Learning](http://www.lab41.org/some-tips-for-debugging-in-deep-learning-2/) - January 2016
- [Faster On-Ramp to Deep Learning With Jupyter-driven Docker Containers](http://www.lab41.org/faster-on-ramp-to-deep-learning-with-jupyter-driven-docker-containers/) - November 2015
- [A Tour of Sentiment Analysis Techniques: Getting a Baseline for Sunny Side Up](http://www.lab41.org/a-tour-of-sentiment-analysis-techniques-getting-a-baseline-for-sunny-side-up/) - November 2015
- [Learning About Deep Learning!](http://www.lab41.org/learning-about-deep-learning/) - September 2015

### Docker Environments
- ```lab41/itorch-[cpu|cuda]```: [iTorch](https://github.com/facebook/iTorch) IPython kernel for [Torch](http://torch.ch/) scientific computing GPU framework
- ```lab41/keras-[cpu|cuda|cuda-jupyter]```: [Keras](http://keras.io/) neural network library (CPU or GPU backend from command line or within Jupyter notebook)
- ```lab41/neon-[cuda|cuda7.5]```: [neon](https://github.com/NervanaSystems/neon) Deep Learning framework (with CUDA backend) by [Nervana](http://www.nervanasys.com/)
- ```lab41/pylearn2```: [pylearn2](https://github.com/lisa-lab/pylearn2) machine learning research library
- ```lab41/sentiment-ml```: build word vectors (Word2Vec from [gensim](https://radimrehurek.com/gensim/); GloVe from [glove-python](https://github.com/maciejkula/glove-python)), tokenize Chinese text ([jieba](https://github.com/fxsjy/jieba) and [pypinyin](https://github.com/mozillazg/python-pinyin)), and tokenize Arabic text ([NLTK](http://www.nltk.org/api/nltk.tokenize.html) and [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml))
- ```lab41/mechanical-turk```: convert CSV of Arabic tweets to individual PNG images for each Tweet (to avoid machine-translation of text) and auto-submit/score Arabic sentiment survey via AWS Mechanical Turk

### Binary Classification with Word Vectors
#### Execution
```python -m benchmarks/baseline_classifiers```

#### Word Vector Models
| model | filename | filesize | vocabulary | details |
|-------|----------|----------|------------|---------|
|[Sentiment140](http://help.sentiment140.com/for-students/) | sentiment140_800000.bin | 153M | 83,586 | [gensim](https://radimrehurek.com/gensim/models/word2vec.html) Word2Vec(size=200, window=5, min_count=10)
|[Open Weiboscope](http://weiboscope.jmsc.hku.hk/datazip/) | openweibo_fullset_hanzi_CLEAN_vocab31357747.bin | 56G | 31,357,746 | [jieba](https://github.com/fxsjy/jieba)-tokenized Hanzi Word2Vec(size=200, window=5, **min_count=1**) |
| Open Weiboscope | openweibo_fullset_min10_hanzi_vocab2548911.bin | 4.6G | 2,548,911 | jieba-tokenized Hanzi Word2Vec(size=200, window=5, **min_count=10**) |
| Arabic [Tweets](https://dev.twitter.com/streaming/public) | arabic_tweets_min10vocab_vocab1520226.bin | 1.2G | 1,520,226 | **[Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml)-tokenized** Word2Vec(size=200, window=5, min_count=10)
| Arabic Tweets | arabic_tweets_NLTK_min10vocab_vocab981429.bin | 759M | 981,429 | **[NLTK](http://www.nltk.org/api/nltk.tokenize.html)-tokenized** Word2Vec(size=200, window=5, min_count=10)

#### Training and Testing Data
| train/test set | filename | filesize | details |
|----------------|----------|----------|---------|
| Sentiment140 | sentiment140_800000_samples_[test/train].bin | 183M | 80/20 split of 1.6M emoticon-labeled Tweets |
| Open Weiboscope | openweibo_hanzi_censored_27622_samples_[test/train].bin | 25M | 80/20 split of 55,244 censored posts
| Open Weiboscope | openweibo_800000_min1vocab_samples_[test/train].bin | 564M | 80/20 split of 1.6M deleted posts
| Arabic Twitter | arabic_twitter_1067972_samples_[test/train].bin | 912M | 80/20 split of 2,135,944 emoticon-and-emoji labeled Tweets

### Binary Classification via Deep Learning
#### CNN (Convolutional Neural Network)
Character-by-character processing From Zhang and LeCun's [Text Understanding From Scratch](http://arxiv.org/pdf/1502.01710v4.pdf):
```python
#Set Parameters for final fully connected layers
fully_connected = [1024,1024,1]

model = Sequential()

#Input = #alphabet x 1014
model.add(Convolution2D(256,67,7,input_shape=(1,67,1014)))
model.add(MaxPooling2D(pool_size=(1,3)))

#Input = 336 x 256
model.add(Convolution2D(256,1,7))
model.add(MaxPooling2D(pool_size=(1,3)))

#Input = 110 x 256
model.add(Convolution2D(256,1,3))

#Input = 108 x 256
model.add(Convolution2D(256,1,3))

#Input = 106 x 256
model.add(Convolution2D(256,1,3))

#Input = 104 X 256
model.add(Convolution2D(256,1,3))
model.add(MaxPooling2D(pool_size=(1,3)))

model.add(Flatten())

#Fully Connected Layers

#Input is 8704 Output is 1024
model.add(Dense(fully_connected[0]))
model.add(Dropout(0.5))
model.add(Activation('relu'))

#Input is 1024 Output is 1024
model.add(Dense(fully_connected[1]))
model.add(Dropout(0.5))
model.add(Activation('relu'))

#Input is 1024 Output is 1
model.add(Dense(fully_connected[2]))
model.add(Activation('sigmoid'))

#Stochastic gradient parameters as set by paper
sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode="binary")
```

#### LSTM (Long Short Term Memory)
```python
# initialize the neural net and reshape the data
model = Sequential()
model.add(Embedding(max_features, embedding_size)) # embed into dense 3D float tensor (samples, maxlen, embedding_size)
model.add(Reshape(1, maxlen, embedding_size)) # reshape into 4D tensor (samples, 1, maxlen, embedding_size)

# convolution stack
model.add(Convolution2D(nb_feature_maps, nb_classes, filter_size_row, filter_size_col, border_mode='full')) # reshaped to 32 x maxlen x 256 (32 x 100 x 256)
model.add(Activation('relu'))

# convolution stack with regularization
model.add(Convolution2D(nb_feature_maps, nb_feature_maps, filter_size_row, filter_size_col, border_mode='full')) # reshaped to 32 x maxlen x 256 (32 x 100 x 256)
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2))) # reshaped to 32 x maxlen/2 x 256/2 (32 x 50 x 128)
#model.add(Dropout(0.25))

# convolution stack with regularization
model.add(Convolution2D(nb_feature_maps, nb_feature_maps, filter_size_row, filter_size_col)) # reshaped to 32 x 50 x 128
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2))) # reshaped to 32 x maxlen/2/2 x 256/2/2 (32 x 25 x 64)
#model.add(Dropout(0.25))

# fully-connected layer
model.add(Flatten())
model.add(Dense(nb_feature_maps * (maxlen/2/2) * (embedding_size/2/2), fully_connected_size))
model.add(Activation("relu"))
model.add(Dropout(0.50))

# output classifier
model.add(Dense(fully_connected_size, 1))
model.add(Activation("sigmoid"))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
```
