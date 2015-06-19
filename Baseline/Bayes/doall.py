from nltk import NaiveBayesClassifier
from import_stanford_twitter import open_stanford_twitter_csv
from feature_evaluator import evaluate_features
from feature_extractors import word_feats
from ingest_twitter import split_tweets

print("Opening CSV file...")
print("Extracting Features...")
path = '/Users/bretts/Documents/Preparing_The_Torch'
path += '/StanfordTweetData/training.1600000.processed.noemoticon.csv'
all_data = open_stanford_twitter_csv(path, feat_extractor=word_feats)
print("CSV file opened and features extracted")
train_set, dev_set, test_set = split_tweets(all_data, train=.9,
                                            dev=0, test=.1, shuffle=True)
print("Data split into sets")
classifier = NaiveBayesClassifier.train(train_set)
print("Classifier trained")
evaluate_features(classifier, test_set)
