import os, re, csv
from os import path
import tarfile
import logging
from urllib2 import urlopen, HTTPError, URLError
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# parameters: data locations
data_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
data_dir = '/data'
data_subfolder_positive = data_dir + '/aclImdb/train/pos/'
data_subfolder_negative = data_dir + '/aclImdb/train/neg/'

# parameters: sentiment scoring
key_text = 'msg'
key_score = 'sentiment'
score_positive = 1
score_negative = 0

# parameters: data modeling
truncate_length = 1014
remove_length = 100
num_entries_max = 10000
vocabulary = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/|_@#$%^&*~`+-=<>()[]{}"


# download and unzip the dataset
def download_and_extract_data(url, download_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.downloads') ):

    # get name and full path to file
    filename = os.path.basename(url)
    file_path = os.path.join(download_dir, filename)

    # download file
    data_download(url, file_path)

    # extract contents
    data_extract(file_path)


# extract data file
def data_extract(file_path):

    download_dir = os.path.dirname(file_path)
    filename = os.path.basename(file_path)

    # extract contents
    logging.info("Unzipping {} in {}...".format(file_path, download_dir)),
    try:
      with tarfile.open(file_path, mode="r:gz") as tar:
        tar.extractall(download_dir)

    # extraction errors
    except IOError, io_error:
        print "IOError:", io_error.code, file_path


# download data file
def data_download(url, file_path):
    ''' downloads file from url to local disk

    @Arguments:
        url           -- the url of the chosen dataset
        download_dir  -- output directory

    @Raises:
        HTTPError, URLError
    '''
    try:
        # prevent re-downloading
        download_dir = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        if os.path.isfile(file_path):
            logging.info("File has already been downloaded")
        else:

          # create download folder if necessary
          if not os.path.exists(download_dir):
              os.mkdir(download_dir)

          # download file to disk
          response = urlopen(url)
          with open(file_path, 'wb') as f:

              # download tarfile
              logging.info("Downloading {} to {}... ".format(filename, download_dir)),
              f.write(response.read())
              logging.info("Success!")

        # return full path to file
        return download_dir

    # Handle errors
    except HTTPError, e:
        print "HTTP Error:", e.code, url
    except URLError, e:
        print "URL Error:", e.reason, url


# load dataframe from disk
def load_dataframe_from_disk(dir_path, score):

    # determine list of files in directory
    files = ["{}{}".format(dir_path,f) for f in os.listdir(dir_path)]

    # create msg-sentiment pairs dataframe from source directory
    df = pd.DataFrame()

    # load files
    cnt = 0
    for filename in files:

        # only load subset
        if cnt > num_entries_max:
            break
        cnt += 1

        with open(filename, 'r') as my_file:
            df = df.append({key_text: my_file.read(), key_score: score}, ignore_index=True)

    # return formatted dataframe
    return df


# download, extract, process, and return a dataframe with positive and negative-labeled messages
def load_dataframe():

    # ensure data is downloaded
    download_dir = download_and_extract_data(data_url, data_dir)

    # load positive and negative messages
    df_pos = load_dataframe_from_disk(data_subfolder_positive, score_positive)
    df_neg = load_dataframe_from_disk(data_subfolder_negative, score_negative)

    # create merged dataset
    df = df_pos.merge(df_neg, how='outer')

    # truncate text
    df[key_text] = df[key_text].str[:truncate_length]

    # remove short entries
    mask = df[key_text].str.len() > remove_length
    df = df[mask]

    # reverse text
    df[key_text] = df[key_text].str[::-1]

    # lowercase text
    df[key_text] = df[key_text].str.lower()

    # return random sample
    return df.sample(num_entries_max, replace=False)


# load data and labels for training and test data in one-hot character-encoding format
def load_character_encoded_data(test_split=.2):

    # load data
    df = load_dataframe()

    # setup the vocabulary for one-hot encoding
    vocab_chars = list(vocabulary)
    vocab_len = len(vocab_chars)

    # convert the vocabulary into a one-hot-encoding
    vocab_hash_table = np.identity(vocab_len)
    df_vocab = DataFrame(vocab_hash_table, columns=vocab_chars)

    # initialize the fixed-size output array
    text_embedding = np.zeros([num_entries_max, truncate_length, vocab_len], dtype='int')

    # create one-hot encoding for each character for each entry in the dataset
    for text_index, text in enumerate(df[key_text][:num_entries_max]):

        # create the output list
        chars = list(text)

        # process text
        for char_index, c in enumerate(chars):
            if(c in vocab_chars):
                text_embedding[text_index][char_index] = np.array(df_vocab[c])

    # Split the text into two groups based on the specified split
    train_test_demarcation = int(len(text_embedding)*(1-test_split))
    X_train = text_embedding[:train_test_demarcation]
    Y_train = df[key_score][:train_test_demarcation]
    X_test = text_embedding[train_test_demarcation:]
    Y_test = df[key_score][train_test_demarcation:]

    # return the train/test sets
    return (X_train, Y_train), (X_test, Y_test)
