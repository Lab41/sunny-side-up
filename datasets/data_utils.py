import os
import csv
import logging
import random
import copy
from urllib2 import urlopen, HTTPError, URLError


def latin_csv_reader(csv_data, dialect=csv.excel, **kwargs):
    ''' Function that takes an opened CSV file with
        ASCII or UTF-8 encoding and convert's to Latin-1

        @Arguments:
            csv_data -- a CSV file opened for reading

            dialect -- specifies the file dialect type

            **kwargs -- other typical arguments one would pass
                into the csv.reader() function.
                Ex: delimiter=','

        @Return:
            A python generator over the lines in the given
            CSV file
    '''

    # Opens CSV reader
    csv_reader = csv.reader(csv_data, dialect=dialect, **kwargs)
    # Converts row to latin-1 encoding and yields this line on next() calls
    for row in csv_reader:
        yield [unicode(cell, 'latin-1') for cell in row]


def get_file(url):
    try:
        # Prevents redownloading
        fname = os.path.join(os.getcwd(), '.downloads', url.split('/')[-1])
        if '.downloads' in os.listdir('.'):
            if url.split('/')[-1] in os.listdir('./.downloads'):
                print("File has already been dowloaded")
                return fname

        # Create hidden folder to hold zip file
        if not os.path.exists(os.path.join(os.getcwd(), '.downloads')):
            os.mkdir(os.path.join(os.getcwd(), '.downloads'))

        response = urlopen(url)

        # Open dowload file and save locally
        with open(fname, 'wb') as f:
            print("Downloading %s... " % url.split('/')[-1]),
            f.write(response.read())
            print("Success!")
        return fname

    # Handle errors
    except HTTPError, e:
        print "HTTP Error:", e.code, url
    except URLError, e:
        print "URL Error:", e.reason, url


def split_data(data, train=.7, dev=.2, test=.1, shuffle=False):
    ''' Takes mapping of tweet to label and splits data into train, dev, and
        test sets according to proportion given

        @Arguments:
            data -- any list type

            train (optional) -- Proportion of tweets to give the training set

            dev (optional) -- Proportion of tweets to give the development set

            test (optional) -- Proportion of tweets to give the testsing set

            shuffle (optional) -- Boolean value that if True randomly puts the
                data into the various sets, rather than in the order of the file

        @Raises:
            ValueError -- If train + dev + test is not equal to 1

        @Return:
            Tuple containing the three sets of data:
                (train_set, dev_set, test_set)
    '''

    # Deals with issues in Floating point arithmetic
    if train * 10 + dev * 10 + test * 10 != 10:
        raise ValueError("Given set proportions do not add up to 1")

    if shuffle:
        # Deep copy handles case that data contains needed objects
        # data = copy.deepcopy(data)
        random.shuffle(data)

    data_size = len(data)
    train_size = int(train * data_size)
    dev_size = int(dev * data_size)

    # Partition data
    train_set = data[0:train_size]
    dev_set = data[train_size + 1:train_size + dev_size]
    test_set = data[train_size + dev_size + 1:data_size]

    return train_set, dev_set, test_set
