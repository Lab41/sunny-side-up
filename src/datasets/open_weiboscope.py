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
logger.setLevel(logging.DEBUG)

from data_utils import get_file

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

def load_data(file_path=None, verbose=False, which_set='train', train_pct=1.0, rng_seed=None):
    """
    Load data from Open Weiboscope corpus of Sina Weibo posts.

    @Arguments:
        file_path -- path to downloaded, unzipped Open Weiboscope
            data. If this path does not exist or is not given, load_data
            will raise an exception (string)
        which_set -- whether to iterate over train or testing set. You should
            also set train_pct and rng_seed to non-default values if you specify this
            (string)
        train_pct -- what percent of dataset should go to training (rest for test; float)
        rng_seed -- value for seeding random number generator

    @Return:
        a generator over a tuples of review text (unicode) and whether or not 
        the tweet was deleted (bool) and the romanized text of the tweet, if jieba
        and pypinyin are installed

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
                        yield (records_split, BadRecordException("Comma split error on mid={} in"
                                         "file {} (len of record: {})".format(
                                            records_split[0], 
                                            os.path.basename(table_path),
                                            len(records_split))))
            
                    # text on field 6, deleted on field 9. screen out retweets (field 1) 
                    if records_split[1] == '':
                        try:
                            yield (records_split[6], records_split[9] != '', romanize_tweet(records_split[6])) 
                        except:
                            yield (records_split[6], records_split[9] != '') 
                except IndexError:
                    # error handling: could yield exceptions into result, or not
                    #yield (records_split, BadRecordException("Possible unicode error/fields in record"))
                    continue

                except UnicodeEncodeError as e:
                    #yield (records_split, e)
                    continue

                except GeneratorExit:
                    return

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
