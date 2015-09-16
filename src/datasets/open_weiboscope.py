#!/usr/bin/env python

import os
import re
from itertools import izip_longest
import random
import codecs
import csv
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

from data_utils import get_file


vocabulary=ur"""abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'"/\|_@#$%^&*~`+-=<>()[]{}""" + "\n"


def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]

def utf_8_encoder(unicode_csv_data): 
    while True:
        try:
            line = unicode_csv_data.next()
            yield line.encode('utf-8')
        except UnicodeDecodeError:
            yield ""
        except StopIteration:
            raise StopIteration()
        except GeneratorExit:
            break

class BadRecordException(Exception):
    pass
class TextTooShortException(Exception):
    pass

def enforce_length(txt, min_length=None, max_length=None, pad_out=False):
    if min_length is not None:
        if len(txt) < min_length:
            raise TextTooShortException()
    if max_length is not None:
        if len(txt) > max_length:
            # truncate txt (from end)
            return txt[0:max_length]
    if pad_out is True:
        txt = txt +  ' ' * (max_length - len(txt))
    return txt

def load_data(file_path=None, which_set='train', form='onehot', train_pct=1.0, rng_seed=None, min_length=None, max_length=None, pad_out=False):
    """
    Load data from Open Weiboscope corpus of Sina Weibo posts. Options are available for encoding
    of returned text data. 

    @Arguments:
        file_path -- path to downloaded, unzipped Open Weiboscope
            data. If this path does not exist or is not given, load_data
            will raise an exception (string)
        which_set -- whether to iterate over train or testing set. You should
            also set train_pct and rng_seed to non-default values if you specify this
            (string)
        form -- return results in hanzi, pinyin romanization, or one-hot encodings of romanization?
            can take values of 'hanzi', 'pinyin', or 'onehot' (string)
        train_pct -- what percent of dataset should go to training (remainder goes to test)?  (float)
        rng_seed -- value for seeding random number generator
        min_length -- enforce a minimum length, in characters, for the 
            dataset? Counted in hanzi for form='hanzi' and in roman characters 
            for form='pinyin'. Texts that are too short will be excluded. (int)
        max_length -- enforce a maximum length, in characters, for the dataset?
            Counted in hanzi or roman characters as approriate (see above).
            Texts that are too long will be truncated at the end. (int)
        pad_out -- for texts shorter than max_length, should they be padded out
            at the end with spaces?

    @Return:
        a generator over a tuples of review text (unicode or numpy array) and whether or not 
        the tweet was deleted (bool)

    """

    if not os.path.exists(file_path):
        # download repository files and unzip them
        os.makedirs(file_path)
        for remote_path in [ "http://weiboscope.jmsc.hku.hk/datazip/week{}.zip".format(a) for a in [ str(b) for b in range(1, 52) ] ]:
            local_zip = get_file(remote_path, file_path)
            with ZipFile(local_zip) as zf:
                zf.extractall(file_path)

    # get list of weekNN.csv files at file_path
    ow_files = [ os.path.join(file_path, f) for f in os.listdir(file_path) if re.match(r"week[0-9]{,2}\.csv", f) is not None ]
    assert ow_files is not []
    
    # strategy: randomize order of weeks (individual files), sample in order from each week.
    try:
        random.seed(rng_seed)
    except:
        pass
    random.shuffle(ow_files)
    split_on = int(len(ow_files) * train_pct)
    data_sets = {}
    logger.debug("Shuffle order: {}, split on {}".format(ow_files, split_on))
    data_sets['train'], data_sets['test'] = ow_files[:split_on], ow_files[split_on:]
    logger.debug(data_sets)
    for table_path in data_sets[which_set]:
        with codecs.open(table_path, "r", encoding="utf-8") as f:
            logging.debug("In file {}".format(table_path))
            for line in unicode_csv_reader(f):
                try:
                    records_split = line
                    if len(records_split) != 11:
                        raise  BadRecordException("Comma split error on mid={} in"
                                         "file {} (len of record: {})".format(
                                            records_split[0], 
                                            os.path.basename(table_path),
                                            len(records_split)))
            
                    # text on field 6, deleted on field 9. screen out retweets (field 1) 
                    if records_split[1] == '':
                        if form=='hanzi':
                            record_txt, sentiment = enforce_length(
                                records_split[6], min_length, max_length, 
                                pad_out), records_split[9] != '' 
                            yield record_txt, sentiment
                        elif form=='pinyin':
                            record_txt, sentiment = enforce_length(
                                romanize_tweet(records_split[6]), min_length, 
                                max_length, pad_out), records_split[9] != '' 
                            yield record_txt, sentiment
                        elif form=='onehot':
                            record_txt, sentiment = enforce_length(
                                romanize_tweet(records_split[6]), min_length, 
                                max_length, pad_out), records_split[9] != '' 
                            yield text_to_one_hot(record_txt, vocabulary), sentiment

                except TextTooShortException:
                    continue
                except BadRecordException:
                    continue
                except IndexError:
                    continue
                except UnicodeEncodeError:
                    continue

                except GeneratorExit:
                    return

def text_to_one_hot(txt, vocabulary=vocabulary):
    # setup the vocabulary for one-hot encoding
    vocab_chars = set(list(vocabulary))

    # create the output list
    chars = list(txt)
    categorical_chars = pd.Categorical(chars, categories=vocab_chars)
    vectorized_chars = np.array(pd.get_dummies(categorical_chars))
    return vectorized_chars

def romanize_tweet(txt):
    import jieba
    import pypinyin as pyp

    def segment_hanzi(txt):
        tokens = jieba.tokenize(txt)
        tokens_hanzi = [tkn[0] for tkn in tokens]
        return tokens_hanzi

    def hanzi_to_pinyin(txt):
        pinyin = pyp.lazy_pinyin(txt, style=pyp.TONE2)
        return u''.join(pinyin)

    hanzi_wds = segment_hanzi(txt)
    pinyin_wds = [ hanzi_to_pinyin(word_chars) for word_chars in hanzi_wds ]
    return u' '.join(pinyin_wds)
