import logging
from data_utils import latin_csv_reader, get_file
import zipfile

# Dictionary that defines the Sentiment features
Sentiment = {0: 'neg', 2: 'neutral', 4: 'pos'}


def load_data(file_path=None, feat_extractor=None, verbose=False):
    ''' Function that takes in a path to the StanfordTweetData CSV
        file, opens it, and adds tweet strings and their respective
        sentiments to a list

        @Arguments:
            file_path -- (optional) personal system file path to the
                training.1600000.processed.noemoticon.csv data set (or others of
                a similar structure)

                The Stanford Sentiment140 Dataset is of the following format per row:
                    [polarity, tweet id, tweet date, query, user, tweet text]

            feat_extractor -- (optional) a function that converts a tweet text string
                and outputs a dictionary of features

            verbose -- if True, this funciton shows logging data as it progresses

        @Return:
            A list of tuples of the following format:
                (tweets/features, sentiment label)
    '''
    tweet_to_sentiment = list()

    # Open file path
    if file_path:
        twitter_csv = open(file_path, 'r')
    else:
        file_path = get_file("http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip")

        with zipfile.ZipFile(file_path, 'r') as zp:
            twitter_csv = zp.open('training.1600000.processed.noemoticon.csv')

    # Perform parsing of CSV file
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
    twitter_csv.close()
    return tweet_to_sentiment
