import csv
import logging
from data_utils import latin_csv_reader, get_file
from io import StringIO
import zipfile

# Dictionary that defines the Sentiment features
Sentiment = {0: 'neg', 2: 'neutral', 4: 'pos'}


def load_data(file_path=None, feat_extractor=None, verbose=False):
    ''' Function that takes in a path to the StanfordTweetData CSV
        file, opens it, and adds tweet strings and their respective
        sentiments to a list

        @Arguments:
            file_path -- personal system file path to the
                training.1600000.processed.noemoticon.csv data set (or others of
                a similar structure)

                The Stanford Sentiment140 Dataset is of the following format per row:
                    [polarity, tweet id, tweet date, query, user, tweet text]

            feat_extractor -- a function that converts a tweet text string
                and outputs a dictionary of features

        @Return:
            A list of tuples of the following format:
                (tweets/features, sentiment label)
    '''
    tweet_to_sentiment = list()
    if file_path:
        with open(file_path, 'r') as twitter_csv:
            # Open CSV file for reading in 'latin-1'
            reader = latin_csv_reader(twitter_csv, delimiter=',')
            for index, tweet in enumerate(reader):
                if verbose and index % 10000 == 0:
                    logging.info("PROGRESS: at tweet #%s", index)

                # Gets tweets string from line in csv
                tweet_string = tweet[5]
                # Gets feature from Sentiment dictionary
                sent = Sentiment[int(tweet[0])]
                # If a feat_extractor function was provided, apply it to tweet
                if feat_extractor:
                    features = feat_extractor(tweet_string)
                    tweet_to_sentiment.append((features, sent))
                else:
                    tweet_to_sentiment.append((tweet_string, sent))
        return tweet_to_sentiment
    else:
        fpath = get_file("http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip")
        with open(fpath, 'rb') as f:
            zfile = zipfile.ZipFile(f)
            data = StringIO.StringIO(zfile.read('trainingandtestdata/training.1600000.processed.noemoticon.csv'))
            reader = latin_csv_reader(data, delimiter=',')
            for index, tweet in enumerate(reader):
                if verbose and index % 10000 == 0:
                    logging.info("PROGRESS: at tweet #%s", index)

                # Gets tweets string from line in csv
                tweet_string = tweet[5]
                # Gets feature from Sentiment dictionary
                sent = Sentiment[int(tweet[0])]
                # If a feat_extractor function was provided, apply it to tweet
                if feat_extractor:
                    features = feat_extractor(tweet_string)
                    tweet_to_sentiment.append((features, sent))
                else:
                    tweet_to_sentiment.append((tweet_string, sent))
        return tweet_to_sentiment
