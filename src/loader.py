import os, sys
import json
import requests
import csv
import tarfile
from os.path import join,exists
import re
import random
from zipfile import ZipFile
import shutil

from datasets.data_utils import get_file, latin_csv_reader
from datasets import sentiment140

# Adds ability to import loader, preprocess
currentPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, currentPath)

import preprocess

cacheDir = "%s/.cached_data" % currentPath


# overall labels
label_positive = 1
label_neutral = 0
label_negative = -1

# sentiment140 download parameters
url_sentiment140 = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
dir_tmp_sentiment140 = '/tmp/sentiment140'
csv_sentiment140 = 'training.1600000.processed.noemoticon.csv'
index_sentiment140_label = 0
index_sentiment140_text = 5
label_sentiment140_positive = '4'
label_sentiment140_negative = '0'

map_amazon_label = {  1.0: label_negative,
                      2.0: label_negative,
                      3.0: label_neutral,
                      4.0: label_positive,
                      5.0: label_positive }


# Filled in at run-time
sizes = {}

# Borrowed from http://stackoverflow.com/questions/9629179/python-counting-lines-in-a-huge-10gb-file-as-fast-as-possible
def blockGen(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b

def nlines(ofFile):
    if ofFile not in sizes:

        with open(ofFile) as f:
            sizes[ofFile] = sum(bl.count("\n") for bl in blockGen(f))

    return sizes[ofFile]

def ensureCache():
    if not exists(cacheDir):
        os.mkdir(cacheDir)

def cacheMaker(cachePath):
    with open(cachePath) as f:
        for l in f:
            yield json.loads(l)

def read_sentiment140(sentiment140Path = "/data/sentiment140/sentiment140.csv"):
    """
        Get a generator for the sentiment140 dataset

        @Arguments:
            sentiment140Path: path to load (and/or save) file

        Process:
            Download file if not present
            Cache file as (tweet,label) pairs

        @Return:
            generator to cached file
    """

    senti140Cache = join(cacheDir, "sentiment140.json")


    # create cached file if necessary
    if not exists(senti140Cache):
        ensureCache()

        # request path to file if necessary
        if not sentiment140Path:
            print("Please provide the local path to the sentiment140 dataset: ")
            sentiment140Path = sys.stdin.readline().strip()

        # download the file if it doesn't exist
        if not exists(sentiment140Path):

            # download entire source zipfile from internet
            print("Downloading sentiment140 dataset from Stanford...")
            file_path = get_file(url_sentiment140)

            # save specified CSV from zipfile
            with ZipFile(file_path, 'r') as zp:
                zp.extract(csv_sentiment140, dir_tmp_sentiment140)
                shutil.move(os.path.join(dir_tmp_sentiment140, csv_sentiment140), sentiment140Path)

        # write to cache
        with open(senti140Cache,"w") as cacheFile:
            with open(sentiment140Path) as sentiPath:

                # enumerate over CSV entries
                reader = latin_csv_reader(sentiPath, delimiter=',')
                for i, line in enumerate(reader):

                    # format text
                    text = preprocess.tweet(line[index_sentiment140_label])

                    # generate binary label
                    if line[index_sentiment140_text] == label_sentiment140_positive:
                      label = label_positive
                    else:
                      label = label_negative

                    # write (text,label) pairs
                    cacheFile.write( json.dumps([text, label]) )
                    cacheFile.write("\n")

    return cacheMaker(senti140Cache)




def read_amazon( amazonPath = "/data/amazon/aggressive_dedup.json" ):
    # Review producer for amazon
    amazonCache = join(cacheDir, "amazon.json")

    # generate cached file if necessary
    if not exists(amazonCache):
        if not amazonPath or not exists(amazonPath):
            print("Please provide the local path to the amazon reviews dataset: ")
            amazonPath = sys.stdin.readline().strip()

        ensureCache()
        with open(amazonCache, "w") as cacheFile:
            with open(amazonPath) as afile:
                for l in afile:
                    jl = json.loads(l)
                    cacheFile.write( json.dumps([jl["reviewText"], map_amazon_label[ jl["overall"] ] ]) )
                    cacheFile.write("\n")

    return cacheMaker(amazonCache)

def read_imdb(imdbPath = "/data/aclImdb/aclImdb_v1.tar"):

    imdbCache = join(cacheDir, "imdb.json")

    polNum = {  "pos":  label_positive,
                "neg":  label_negative,
                "unsup":label_neutral }

    if not exists(imdbCache):
        if not imdbPath or not exists(imdbPath):
            print("Please provide the local path to the imdb reviews dataset: ")
            imdbPath = sys.stdin.readline().strip()

        with tarfile.open(imdbPath) as tarf:
            ensureCache()
            with open(imdbCache,"w") as cacheFile:
                for name in tarf.getnames():
                    if name.endswith(".txt"):
                        try:

                            pol = name.split("/")[2]
                            pNum = polNum[pol]
                            review = tarf.extractfile(name).read().strip()
                            cacheFile.write( json.dumps([review, pNum]) )
                            cacheFile.write("\n")

                        except Exception as e:
                            print("Ignoring error on %s" % name)

    return cacheMaker(imdbCache)

def limiter(baseGen, limit):
    assert(limit > 0)
    while limit > 0:
        limit -= 1
        yield( next(baseGen) )

def sampler(baseGen, sampleRate):
    for item in baseGen:
        if random.random() < sampleRate:
            yield( item )

def read(dataset, dataPath=None, limit=None, sampleRate=None):

    # obtain generator over (tweet,label) pairs
    gen = None
    if dataset == "amazon":
        gen = read_amazon(dataPath)
    elif dataset == "sentiment140":
        gen = read_sentiment140(dataPath)
    elif dataset == "imdb":
        gen = read_imdb(dataPath)
    else:
        print("No dataset, %s, your options are: imdb, amazon or sentiment140" % dataset)
        exit()

    # sample data if specified
    if sampleRate:
        gen = sampler(gen, sampleRate)

    # limit maximum results
    if limit:
        gen = limiter(gen, limit)

    # return generator over data
    return gen
