import os, sys
import json
import requests
import csv
import tarfile
from os.path import join,exists
import re
import random

# Adds ability to import loader, preprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import preprocess

cacheDir = "/data/data/.cache"

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
          may need to provide local path to data.
    """

    senti140Cache = join(cacheDir, "sentiment140.json")
    if not exists(senti140Cache):
        ensureCache()
        if not sentiment140Path or not exists(sentiment140Path):
            print("Please provide the local path to the sentiment140 dataset: ")
            sentiment140Path = sys.stdin.readline().strip()

        with open(senti140Cache,"w") as cacheFile:
            with open(sentiment140Path) as sentiPath:
                reader = csv.reader(sentiPath)
                for line in reader:
                    cacheFile.write( json.dumps([preprocess.tweet(line[5].decode("utf-8")), 1 if line[0] == '4' else -1]) )
                    cacheFile.write("\n")
                    
    return cacheMaker(senti140Cache)
    
def read_amazon( amazonPath = "/data/amazon/aggressive_dedup.json" ):
    # Review producer for amazon
    amazonCache = join(cacheDir, "amazon.json")
    if not exists( amazonCache ): 
        if not amazonPath or not exists(amazonPath):
            print("Please provide the local path to the amazon reviews dataset: ")
            amazonPath = sys.stdin.readline().strip()

        reviewLevel = { 1.0 : -1, 2.0: -1, 3.0: 0, 4.0: 1, 5.0 : 1 }
        ensureCache()
        with open(amazonCache, "w") as cacheFile:
            with open(amazonPath) as afile:
                for l in afile:
                    jl = json.loads(l)
                    cacheFile.write( json.dumps([jl["reviewText"], reviewLevel[ jl["overall"] ] ]) )
                    cacheFile.write("\n")
        
    return cacheMaker(amazonCache)

def read_imdb(imdbPath = "/data/aclImdb/aclImdb_v1.tar"):
    
    imdbCache = join(cacheDir, "imdb.json")
    
    polNum = {"pos":1, "neg":-1, "unsup":0}
    
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
                            review = tarf.getfile().read().strip()
                            cacheFile.write( json.dumps([review, pNum]) )
                            cacheFile.write("\n")
                            
                        except IndexError, KeyError:
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
    
    if sampleRate:
        gen = sampler(gen, sampleRate)
    
    if limit:
        gen = limiter(gen, limit)
    
    return gen