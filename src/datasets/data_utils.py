import os, re, csv
import logging
import random
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
    ''' Takes in a file url from the web and dowloads that file to
    to a local directory called ./.downloads

    @Arguments:
        url -- the url of the chosen dataset

    @Raises:
        HTTPError, URLError
    '''
    try:
        # Prevents redownloading
        file_path = os.path.dirname(os.path.abspath(__file__))
        downloads_path = os.path.join(file_path, './.downloads')
        fname = os.path.join(file_path, '.downloads', url.split('/')[-1])
        if '.downloads' in os.listdir(file_path):
            if url.split('/')[-1] in os.listdir(downloads_path):
                logging.info("File has already been dowloaded")
                return fname

        # Create hidden folder to hold zip file
        if not os.path.exists(downloads_path):
            os.mkdir(downloads_path)

        response = urlopen(url)

        # Open dowload file and save locally
        with open(fname, 'wb') as f:
            logging.info("Downloading %s... " % url.split('/')[-1]),
            f.write(response.read())
            logging.info("Success!")
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


def preprocess_tweet(text):
    '''Script for preprocessing tweets by Romain Paulus
    with small modifications by Jeffrey Pennington
    with translation to Python by Motoki Wu

    Translation of Ruby script to create features for GloVe vectors for Twitter data.
    By Motoki Wu
    http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

    @Arguments:
        "Some random text with #hashtags, @mentions and http://t.co/kdjfkdjf (links). :)"

    @Return:
        Processed tweet with certain features replaced

    '''
    FLAGS = re.MULTILINE | re.DOTALL

    # Different regex parts for smiley faces
    eyes = ur"[8:=;]"
    nose = ur"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    def hashtag(text):
        text = text.group()
        hashtag_body = text[1:]
        if hashtag_body.isupper():
            result = u"<hashtag> {} <allcaps>".format(hashtag_body)
        else:
            result = u" ".join([u"<hashtag>"] + re.split(ur"(?=[A-Z])", hashtag_body, flags=FLAGS))
        return result

    def allcaps(text):
        text = text.group()
        return text.lower() + u" <allcaps>"

    text = re_sub(ur"https?:\/\/\S+\b|www\.(\w+\.)+\S*", u"<url>")
    text = re_sub(ur"/", " / ")
    text = re_sub(ur"@\w+", u"<user>")
    text = re_sub(ur"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), u"<smile>")
    text = re_sub(ur"{}{}p+".format(eyes, nose), u"<lolface>")
    text = re_sub(ur"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), u"<sadface>")
    text = re_sub(ur"{}{}[\/|l*]".format(eyes, nose), u"<neutralface>")
    text = re_sub(ur"<3", "<heart>")
    text = re_sub(ur"[-+]?[.\d]*[\d]+[:,.\d]*", u"<number>")
    text = re_sub(ur"#\S+", hashtag)
    text = re_sub(ur"([!?.]){2,}", ur"\1 <repeat>")
    #text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    ## -- I don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(ur"([A-Z]){2,}", allcaps)

    return text.lower()
