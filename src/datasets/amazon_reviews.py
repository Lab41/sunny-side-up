import os
import gzip
import json
import logging
import h5py
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
from data_utils import get_file


class BoringException(Exception):
    pass

def process_amazon_json(json_line):
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
        data, opens it, and yields a dictoray of information for each
        review

        @Arguments:
            file_path -- (optional) personal system file path to the
                SNAP Stanford data set (or others of a similar structure)
            
            amazon_url -- (optional) URI of data set, in case it needs to be 
                downloaded. Defaults to Home and Kitchen reviews
        @Return:
            A generator over a dictionaries of each Amazon Reveiws
    '''

    # Open file path
    if not os.path.isfile(file_path):
        file_path = get_file(amazon_url, os.path.dirname(file_path))

    # Parse Amazon Reviews GZip file
    with gzip.open(file_path, 'r') as f:
        for l in f:
            try:
                review_text, sentiment = process_amazon_json(l)
                yield review_text.decode("latin1"), sentiment
            except BoringException as e:
                logger.info(e)
                continue


                

        

            
            
    
                              
    