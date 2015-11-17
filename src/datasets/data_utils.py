import os, re, csv, errno, sys
import logging
import random
from urllib2 import urlopen, HTTPError, URLError
from nltk.tokenize import wordpunct_tokenize
import numpy as np
import logging
logging.basicConfig()
logger=logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class DataException(Exception):
    pass

class TextTooShortException(DataException):
    pass

class TextIterator:
    '''
        Iterator for text in (text,sentiment) tuples returned by a generator
    '''
    def __init__(self, klass, dirname):
        self.data = klass(dirname).load_data()
        self.counter = 0

    def __iter__(self):
        return self

    def next(self):
        text,sentiment = self.data.next()
        self.counter += 1
        return tokenize(text)


class DataSampler:
    '''
        Data sampling class to load samples from data loader
    '''
    def __init__(self, klass, file_path, num_classes):
        self.klass = klass
        self.file_path = file_path
        self.samples = {}
        self.num_classes = num_classes

    def min_current_samples(self):
        '''
            Return the lowest number of samples of all class types
        '''
        if len(self.samples):
            return min([len(samples) for samples in self.samples.itervalues()])
        else:
            return 0

    def sample_balanced(self, min_samples=None, shuffle=True, sample=False):
        '''
            Returns dataset with equal numbers of each label

            @Arguments:
                min_samples --  if not None, will return dataset with each label having <= this number datapoints
                                if None, will return dataset with each label having a size equal to the smallest subset size

                shuffle     --  if True, will return dataset with shuffled (data,label) tuples
                                if False, will return dataset ordered by (data,label1)...(data,label2)...

                sample      --  if True, will sample from overrepresented datasets after loading
                                if False, will simply return the first-N samples from each label type

            @Returns:
                list of (data, label) tuples
        '''

        # return expected (text,sentiment) tuples
        tuples = []
        self.samples = {}

        # process all samples
        for text,sentiment in self.klass(self.file_path).load_data():

            # calculate minimum samples in each category
            min_current_samples = self.min_current_samples()
            num_classes = len(self.samples)
            if (min_samples and (min_current_samples >= min_samples) and (num_classes == self.num_classes)):
                break
            else:

                # append to list of samples for that type
                try:

                    # append value if sampling or if more values needed
                    if sample or len(self.samples[sentiment]) < min_samples:
                        self.samples[sentiment].append(text)

                # create list of samples for first entry
                except KeyError as e:
                    self.samples[sentiment] = [text]

        # truncate to lowest minimum number (in case total number are less than desired min)
        min_current_samples = self.min_current_samples()

        # process each label type via either sampling or N-first
        for sentiment in self.samples.iterkeys():

            # randomly sample among all possible
            if sample:

                # generate subsample of random indices out of total available
                indices = range(len(self.samples[sentiment]))
                random.shuffle(indices)
                indices_sample = random.sample(indices, min_current_samples)

                # keep entries at those random indices
                for i in indices_sample:
                    tuples.append( (self.samples[sentiment][i],sentiment) )

            # only return
            else:
                for text in self.samples[sentiment][:min_current_samples]:
                    tuples.append( (text,sentiment) )

        # optionally-shuffle tuples when not sampling
        if shuffle:
            random.shuffle(tuples)

        # return results
        return tuples



def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def tokenize(txt):
    return wordpunct_tokenize(txt)


def normalize(txt, vocab=None, replace_char=' ',
                min_length=100, max_length=1014, pad_out=True, 
                to_lower=True, reverse = True, 
                truncate_left=False, encoding="latin1"):
    ''' Takes a single string object and truncates it to max_length,
    raises an exception if its length does not exceed min_length, and
    performs case normalization if to_lower is True. Optionally
    replaces characters not in vocab with replace_char

    @Arguments:
        txt -- a text object to be normalized

        vocab -- an iterable of allowable characters; characters
            out of vocab will be replaced with replace_char

        replace_char -- replace out-of-vocab chars with this

        min_length -- if len(txt) is less than this, raise
            TextTooShortException. Set to None or 0 to disable

        max_length -- txt longer than this will be truncated to
            max_length. Set to None to disable
        
        pad_out -- pad out texts shorter than max_length to max_length,
            using replace_char
    
        to_lower -- if True, all characters in txt wil be
            coerced to lower case
            
        encoding -- if not None, encode txt using this encoding

    @Returns:
        Normalized version of txt

    @Raises:
        DataException, TextTooShortException
    '''
    # reject txt if too short
    if len(txt) < min_length:
        raise TextTooShortException("Too short: {}".format(len(txt)))
    # truncate if too long
    if truncate_left:
        txt = txt[-max_length:]
    else:    
        txt = txt[0:max_length]
    # change case
    if to_lower==True:
        txt = txt.lower()
    # Reverse order
    if reverse == True:
        txt = txt[::-1]
    # replace chars
    if vocab is not None:
        txt = ''.join([c if c in vocab else replace_char for c in txt])
    # re-encode text
    if encoding is not None:
        txt = txt.encode(encoding, errors="ignore")
    # pad out if needed
    if pad_out==True:
        txt = replace_char * (max_length - len(txt)) + txt        
    return txt

zhang_lecun_vocab=list("abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/|_@#$%^&*~`+-=<>()[]{}")
zhang_lecun_vocab_hash = {b: a for a, b in enumerate(zhang_lecun_vocab)}
def to_one_hot(txt, vocab=zhang_lecun_vocab, vocab_hash=zhang_lecun_vocab_hash):
    vocab_size = len(vocab)
    one_hot_vec = np.zeros((vocab_size, len(txt)))
    # run through txt and "switch on" relevant positions in one-hot vector
    for idx, char in enumerate(txt):
        try:
            vocab_idx = vocab_hash[char]
            one_hot_vec[vocab_idx, idx] = 1
        # raised if character is out of vocabulary
        except KeyError:
            pass
    return one_hot_vec

def from_one_hot(oh, vocab=zhang_lecun_vocab):
    vocab_hash = { a: b for a, b in enumerate(vocab) }
    txt = []
    for col_i in range(oh.shape[1]):
        if oh[:,col_i].sum() > 0:
            txt.append(vocab_hash[oh[:, col_i].argmax()])
        else:
            txt.append(" ")
    return txt

def index_as_one_hot(indices, axis=0, dtype=np.int32):
    encoding_size = np.max(indices)
    encoding_length = len(indices)
    encoded_indices = []
    for index in indices:
        encoded_index = np.zeros(encoding_length, dtype)
        encoded_index[index] = 1
        encoded_indices.append(encoded_index)
    onehot_encoding = np.concatenate(encoded_indices, axis=axis)
    return onehot_encoding

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


def get_file(url, dest_path="./downloads"):
    ''' Takes in a file url from the web and dowloads that file to
    to a directory given in dest_path

    @Arguments:
        url -- the url of the chosen dataset
        file_path -- the directory in which the hidden downloads folder 
            is created. (defaults to pwd)

        dest_path -- the destination path

    @Raises:
        HTTPError, URLError
    '''
    try:
        # Prevents redownloading
        #file_path = os.path.dirname(os.path.abspath(__file__))
        #downloads_path = os.path.join(file_path, './.downloads')
        fname = os.path.join(dest_path, url.split('/')[-1])
        #if '.downloads' in os.listdir(file_path):
        #    if url.split('/')[-1] in os.listdir(downloads_path):
        #        logging.info("File has already been dowloaded")
        #        return fname
        if os.path.exists(fname):
            return fname

        # Create hidden folder to hold zip file
        if not os.path.exists(dest_path):
            os.mkdir(dest_path)

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


# simple structure to hold results of profiled methods
class ProfileResults():
    def __init__(self):
        self.results = None
        self.timer = None

# decorator to profile execution time
def timed(func):
    import cProfile, pstats
    def func_wrapper(*args, **kwargs):
        # return object
        profile_results = ProfileResults() #{ 'results': None, 'timer': None }

        # initialize profiler
        pr = cProfile.Profile()
        pr.enable()

        # execute method
        profile_results.results = func(*args, **kwargs)

        # profile results
        pr.disable()
        ps = pstats.Stats(pr)
        profile_results.timer = ps

        # return object with results and timer
        return profile_results

    return func_wrapper


def syslogger(name="logger"):

    # setup stdout logger
    log = logging.getLogger(name)
    out_hdlr = logging.StreamHandler(sys.stdout)

    # set formatting
    out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))

    # set default level to info
    out_hdlr.setLevel(logging.INFO)
    log.addHandler(out_hdlr)
    log.setLevel(logging.INFO)

    # return logger
    return log
