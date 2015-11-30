import os
import logging
import shutil
import random
logger = logging.getLogger(__name__)
from data_utils import latin_csv_reader, get_file
from zipfile import ZipFile

# Dictionaries that define the Sentiment features
sentiment_text = {0: 'neg', 2: 'neutral', 4: 'pos'}
sentiment_binary = {0: 0, 4: 1}


# backwards-compatibility
def load_data(file_path="./.downloads/sentiment140.csv", feat_extractor=None, verbose=False, return_iter=True, rng_seed=None):
    loader = Sentiment140(file_path)
    return loader.load_data(feat_extractor=feat_extractor, verbose=verbose, return_iter=return_iter, rng_seed=rng_seed)

def to_txt(write_path, read_path=None, verbose=False):
    loader = Sentiment140(read_path)
    return loader.to_txt(write_path, verbose=verbose)


class Sentiment140:

    def __init__(self, file_path="./.downloads/sentiment140.csv"):

        # download the data
        dir_root = self.download_data(file_path)

        # gauge size of data
        self.samples = 0

        # get the data and shuffle it
        self.file_path = file_path


    def num_samples(self):
        return self.samples


    def download_data(self, file_path):

        # download file
        if not os.path.isfile(file_path):

            # download and save file from internet
            logger.info("Downloading {}...".format(file_path))
            file_downloaded = get_file("http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip")

            # extract csv
            filename = 'training.1600000.processed.noemoticon.csv'
            file_dir = os.path.dirname(file_path)
            with ZipFile(file_downloaded, 'r') as zp:
                zp.extract(filename, path=file_dir)
                shutil.move(os.path.join(file_dir, filename), file_path)


    def load_data(self, feat_extractor=None, verbose=False, return_iter=True, rng_seed=None):
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

                return_iter -- if True, return an iterator over tuples of (record, sentiment);
                   if False, return a list of such tuples

            @Return:
                A list of tuples of the following format:
                    (tweets/features, sentiment label)
        '''
        tweet_to_sentiment = list()

        # Open file path
        try:
            twitter_csv = open(self.file_path, 'r')
        except IOError as e:
            logger.exception("File I/O error, will try downloading...")
            logger.info("Downloading...")
            self.download_data(self.file_path)
            twitter_csv = open(self.file_path, 'r')


        # Perform parsing of CSV file
        reader = latin_csv_reader(twitter_csv, delimiter=',')
        for i, tweet in enumerate(reader):
            # Prints progress every 10000 words read
            if verbose and i % 10000 == 0:
                logging.info("PROGRESS: at tweet #%s", i)

            # Gets tweets string from line in csv
            tweet_string = tweet[5]


            # ensure feature is in Sentiment dictionary
            try:
                sent = sentiment_binary[int(tweet[0])]

                # If a feat_extractor function was provided, apply it to tweet
                if feat_extractor:
                    features = feat_extractor(tweet_string)
                    tweet_to_sentiment.append((features, sent))
                else:
                    tweet_to_sentiment.append((tweet_string, sent))

                # tally number of samples
                self.samples += 1

            except KeyError:
                logger.debug("Sentiment score of {} skipped.".format(tweet[0]))

        twitter_csv.close()

        # shuffle dataset
        random.seed(rng_seed)
        random.shuffle(tweet_to_sentiment)

        # return list or iterator
        if return_iter:
            return iter(tweet_to_sentiment)
        else:
            return tweet_to_sentiment


    def to_txt(write_path, read_path=None, verbose=False):
        ''' Function that takes in a path to the StanfordTweetData CSV
            file, opens it, and writes the tweets with new lines to an output
            file.
        '''
        read_path = self.load_data(verbose=verbose)
        with open(read_path, 'r') as twitter_csv, open(write_path, 'w') as output:
            reader = latin_csv_reader(twitter_csv, delimiter=',')
            # For each line in CSV, write each tweet with a new line to the output
            for line in reader:
                output.write(line[5].encode('UTF-8') + '\n')

def main():
    # Download data (will save in ./.downloads)
    data = load_data()

if __name__=="__main__":
    main()
