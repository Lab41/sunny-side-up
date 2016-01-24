#!/usr/bin/env python
import os, sys
import re
from itertools import izip_longest
import random
import csv
import logging
import numpy as np
import data_utils
from zipfile import ZipFile
from data_utils import get_file, to_one_hot
import jpype
import glob

# update field size
csv.field_size_limit(sys.maxsize)

# setup logging
logger = data_utils.syslogger(__name__)

class BadRecordException(Exception):
    pass
class TextTooShortException(Exception):
    pass



class ArabicTwitter:

    def __init__(self, file_path):
        self.samples = 0
        self.file_path = file_path

        # initialize jvm with stanford nlp
        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), '-ea', '-Djava.class.path=/data/stanford-segmenter-2015-04-20/stanford-segmenter-3.5.2.jar:%s/stanford-parser.jar' % os.environ["STANFORD_PARSER_HOME"])

        # import java base classes
        self.String = jpype.java.lang.String
        self.StringReader = jpype.JClass('java.io.StringReader')
        self.BufferedReader = jpype.JClass('java.io.BufferedReader')
        self.InputStreamReader = jpype.JClass('java.io.InputStreamReader')
        self.StringBufferInputStream = jpype.JClass("java.io.StringBufferInputStream")

        # import java stanford-nlp classes
        self.tokenizerFactory = jpype.JClass("edu.stanford.nlp.international.arabic.process.ArabicTokenizer").factory()

    def load_data_raw(self, file_path=None):

        # set default file path
        if file_path is None:
            file_path = self.file_path

        # use regex to find multiline tweets
        # memory inefficient, but captures multiline tweets
        regex = re.compile("TWEET123START(.*?)TWEET789END", re.DOTALL)
        with open(self.file_path, 'r') as f:
            contents = f.read()
            lines = regex.findall(contents)

        # generate over lines
        for line in lines:
            yield line

    def tokenize_arabic(self, text):

        # default if errors
        tokens = ['']

        # attempt to tokenize
        try:
            # parser needs unicode literal
            text_u = unicode(text, 'raw_unicode_escape')

            # tokenize input
            bufferedReader = self.BufferedReader(self.InputStreamReader(self.StringBufferInputStream(self.String(text_u)), "UTF-8"))
            line = bufferedReader.readLine()
            tokenizedLine = self.tokenizerFactory.getTokenizer(self.StringReader(line)).tokenize()

            # return utf-8 encoded tokens
            tokens = [tok.toString().encode('utf-8') for tok in tokenizedLine]

        except UnicodeDecodeError as e:
            logger.debug(e)
        except Exception as e:
            logger.debug(e)

        # return tokens or default empty list
        return tokens

    def twitter_strip(self, text):

        # positive and negative emoticons from initial twitter search
        emoticons_pos = [':)', ':-)', ':D', '=)', '\xF0\x9F\x8E\x81', '\xE2\x9D\xA4', '\xF0\x9F\x92\x9E', '\xF0\x9F\x92\x83', '\xF0\x9F\x8E\x8A'] #, '\xF0\x9F\x8E\x89', '\xF0\x9F\x92\x99', '\xF0\x9F\x98\x9A', '\xF0\x9F\x92\x96', '\xF0\x9F\x98\x98']
        emoticons_neg = [':(', ':-(', ':\'(', '=(', '\xF0\x9F\x98\x95', '\xF0\x9F\x98\xA9', '\xF0\x9F\x98\x92', '\xF0\x9F\x98\xA0', '\xF0\x9F\x98\xA1'] #, '\xF0\x9F\x98\x91', '\xF0\x9F\x94\xAB', '\xF0\x9F\x98\xAB', '\xF0\x9F\x98\x9E', '\xF0\x9F\x98\xA4']

        # strip retweets
        stripped = text
        stripped = re.sub("RT @.*?: ",'',stripped)

        # strip emoticons
        for emoticon in emoticons_pos + emoticons_neg:
            stripped = stripped.replace(emoticon,'')

        # return stripped text
        return stripped

    def load_data(self):
        """
        Load data from Arabic Twitter corpus from public API.

        @Arguments:
            file_path -- path to JSON-formatted Twitter messages

        @Return:
            a generator over a tuples of text (unicode or numpy array) and sentiment
        """

        # use regex to find multiline (id,tweet,sentiment)
        regex = re.compile("(\d{18}),(.*?),([0|1])\n", re.DOTALL)

        # iterate all files in specified directory
        for file_path in glob.glob(os.path.join(self.file_path, '*')):
            with open(file_path, 'r') as f:
                contents = f.read()
                lines = regex.findall(contents)

                for tweet_id, tweet, sentiment in lines:
                    yield tweet, int(sentiment)


class ArabicTwitterIterator:
    '''
        Iterator for text in (text,sentiment) tuples returned by a generator
    '''
    def __init__(self, file_path):
        self.counter = 0
        self.arabic_data = ArabicTwitter(file_path)
        self.data = self.arabic_data.load_data_raw(file_path)

    def __iter__(self):
        return self

    def next(self):
        text = self.data.next()

        # increment and output progress of counter
        self.counter += 1
        if self.counter % 10000 == 0:
            print("Iterator at {}".format(self.counter))

        # return tokenized text
        return self.arabic_data.tokenize_arabic(text)
