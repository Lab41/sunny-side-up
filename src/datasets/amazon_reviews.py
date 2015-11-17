import os
import gzip
import json
import logging
import h5py
import shutil
from data_utils import get_file, syslogger
logger = syslogger(__name__)


class BoringException(Exception):
    pass


class AmazonReviews:


    def __init__(self, file_path='/data/amazon/reviews_Home_and_Kitchen.json.gz',
                        amazon_url =   "http://snap.stanford.edu/data/amazon/"
                                       "productGraph/categoryFiles/"
                                       "reviews_Home_and_Kitchen.json.gz"):

        # download the data if necessary
        self.file_path = file_path
        data_root = self.download_data(file_path, amazon_url)

        # initialize the number of samples
        self.samples = 0



    def download_data(self, file_path='/data/amazon/reviews_Home_and_Kitchen.json.gz',
                            amazon_url =   "http://snap.stanford.edu/data/amazon/"
                                           "productGraph/categoryFiles/"
                                           "reviews_Home_and_Kitchen.json.gz"):
        # download data if necessary
        filename_url = os.path.basename(amazon_url)
        dir_data = os.path.dirname(file_path)
        if not os.path.isfile(file_path):
            file_downloaded = get_file(amazon_url, dir_data)
            shutil.move(os.path.join(dir_data, filename_url), file_path)

        # return parent data directory
        return dir_data



    def num_samples(self):
        return self.samples


    def load_data(self):

        # Parse Amazon Reviews GZip file
        with gzip.open(self.file_path, 'r') as f:
            for l in f:
                try:
                    review_text, sentiment = self.process_amazon_json(l)
                    yield review_text.decode("latin1"), sentiment
                except BoringException as e:
                    #logger.info(e)
                    continue

    def process_amazon_json(self, json_line):
        '''
        Ingest a single JSON recor of an Amazon review. Return
        text and rating. Raises an exception when the review has a
        3 star rating
        '''
        json_obj = json.loads(json_line)

        if json_obj['overall'] == 3.0:
            raise BoringException("Boring review")
        elif json_obj['overall'] < 3.0:
            overall = 0
        else:
            overall = 1

        return json_obj['reviewText'], overall



def load_data(file_path='/data/amazon/reviews_Home_and_Kitchen.json.gz',
              amazon_url = "http://snap.stanford.edu/data/amazon/"
                           "productGraph/categoryFiles/"
                           "reviews_Home_and_Kitchen.json.gz"):
    ''' Function that takes in a path to the Stanford SNAP Amazon review
        data, opens it, and yields a tuple of information for each
        review

        @Arguments:
            file_path -- personal system file path to the
                SNAP Stanford data set (or others of a similar structure)

            amazon_url -- (optional) URI of data set, in case it needs to be
                downloaded. Defaults to Home and Kitchen reviews
        @Return:
            A generator over a dictionaries of each Amazon Reveiws
    '''

    amazon_reviews = AmazonReviews(file_path, amazon_url)

    return amazon_reviews.load_data()


if __name__=="__main__":
    data = load_data()
    print data.next()
