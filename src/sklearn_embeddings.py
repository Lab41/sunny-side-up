import os, sys
import argparse
from collections import defaultdict, Counter


# Word2Vec/Doc2Vec packages
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords as stopCorpus

import random
import logging
import numpy as np
from math import ceil

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import math
from itertools import izip, imap

# Adds ability to import loader, preprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import loader, preprocess
from glove import Glove, Corpus

englishStops = set( stopCorpus.words("english") )

def mapNGrams(words, n):
    nwords = len(words)
    for i in xrange(2,n+1):
        for j in xrange(i-1, nwords):
            words.append(" ".join(words[j-i : j]))
        
def getFolds(allData, k=5):
    perFoldLen = int(ceil(len(allData) / float(k)))
    
    yield allData[perFoldLen:], allData[:perFoldLen]
    
    for i in range(1,k-1):
        yield allData[:perFoldLen*i] + allData[perFoldLen*(i+1):], allData[perFoldLen*i : perFoldLen*(i+1)]
        
    yield allData[: -perFoldLen], allData[-perFoldLen:]
        
def build_lsi_embeddings(training,testing,args):
    raise NotImplementedError("Coming soon...")
        
def build_glove_embeddings(training, testing, args):
    
    ''' Trains the model on the sentiment140 dataset

    @Arguments:
        data:  the loaded sentiment140 dataset from module
        num_epochs: the number of epochs to train on
        num_threads: the number of threads to use
        num_components: the number of components the glove model should use
        learning_rate: the model's learning rate
        window_size: the size of the window to use when looking for word co-occurence
        verbose: boolean for whether or not extensive output should be printed to screen

    @Return:
        A trained glove model
    '''
        
    # initialize model
    glove = Glove(no_components = args.vecsize, learning_rate = args.learningRate)
    
    # read in the data to train on
    corpus_model = Corpus()
    corpus_model.fit( imap(preprocess.tokenize, training), window = args.window)
        
    # fit the model using the given parameters
    glove.fit(corpus_model.matrix, epochs = args.epochs, no_threads = args.parallelism, verbose = args.verbose)
              
    # add a dictionary just to make it easier for similarity queries
    glove.add_dictionary(corpus_model.dictionary)
    
    transformer = lambda words: model.transform_paragraph(words, use_pca = use_pca)

    fromTraining = to_sklearn_format(transformer, training, args.vecsize)
    fromTesting = to_sklearn_format(transformer, testing, args.vecsize)
    
    return fromTraining, fromTesting

    return glove
        
def build_doc2vec_embeddings(training, testing, args):
    ''' Trains the Doc2Vec on the sentiment140 dataset

    @Arguments:
        data -- the loaded sentiment140 dataset from module

        epoch_num -- sets the number of epochs to train on

    @Return:
        A trained Doc2Vec model
    '''
    labeled_sent = []
    
    count = 0
    ls = None

    cleaned = lambda v: filter(lambda w: w not in englishStops, v)
    
    ''' Sets the label for each individual sentence in the Doc2Vec model.
    These become "special words" that allow the vector for a sentence to
    be accessed from the model. Each label must be unique '''
    for (sentence, label) in training:
        ls = TaggedDocument(cleaned( sentence.split() ), ['lbl%d_%d' % (label, count)])
        labeled_sent.append(ls)
    
    logging.info("Building model...")

    '''Setting min_count > 1 can cause some tweets to "disappear" later
    from the Doc2Vec sentence corpus.
    ex: you could imagine a tweet containing only words whose count was low'''
    model = Doc2Vec(min_count=args.dvMinCount,
                    window=args.window,
                    size=args.vecsize,
                    sample=args.dvSample,
                    negative=args.dvNegative,
                    workers=args.dvWorkers)

    logging.info("Building Vocabulary...")
    model.build_vocab(labeled_sent)

    logging.info("Training model...")
    for epoch in xrange(args.epochs):
        logging.info("Epoch %s..." % epoch)
        
        # Numpy random permutation method shuffles data in place 
        # Shuffling improves the accuracy of the model
        random.shuffle(labeled_sent)
        logging.getLogger().setLevel(logging.WARN)
        model.train( labeled_sent )
        logging.getLogger().setLevel(logging.INFO)
        
    fromTraining = to_sklearn_format(model.infer_vector, training, args.vecsize)
    fromTesting = to_sklearn_format(model.infer_vector, testing, args.vecsize)
    
    return fromTraining, fromTesting


def to_sklearn_format(converter, rawDocs, vecsize):
    ''' Convert docs into vectors using model

    @Arguments:
        model -- A trained and loaded Doc2Vec model of Sentiment140

        taggedDocs -- List of TaggedDocument objects

    @Return:
        tuple of 2 numpy arrays consisting of the data and labels
    '''
    
    # Initializes numpy data matrices and label vectors
    data = np.zeros((len(rawDocs), vecsize))
    labels = np.zeros(len(rawDocs))
    
    i=0
    for (doc, lbl) in rawDocs:
        
        data[i,:] = converter(filter(lambda w: w not in englishStops, doc.split()) )
        labels[i] = lbl
        i += 1
        
    return data, labels

def getClassifiers(args):
    """
        Returns a list of classifier tuples (name, model)
        for use in training
    """
    return [("LogisticRegression", LogisticRegression(C=1.0,
                                                      class_weight=None,
                                                      dual=False,
                                                      fit_intercept=True,
                                                      intercept_scaling=1,
                                                      penalty='l2', 
                                                      random_state=None,
                                                      tol=0.0001),),
            
           ("RandomForests", RandomForestClassifier(n_jobs=-1,
                                                    n_estimators = args.nTrees,
                                                    max_features = args.rfFeatures)),
           ("Gaussian NaiveBayes", GaussianNB(),)]


def test_model(full_dataset, args):
    ''' Uses a loaded Doc2Vec model and a LogisticRegression
    from the scikitlearn package to build a sentiment classifier

    @Argument:
        model -- A trained and loaded Doc2Vec model of Sentiment140
        full_dataset -- All documents
    '''
    dist = Counter()
    f1Scores = defaultdict(list)
    accuracyScores = defaultdict(list)
    i=0
    # Converts data to Sklearn acceptable numpy format
    for training,testing in getFolds(full_dataset, args.k):
        i += 1
        logging.info("Working on fold %s" % i)
        
        if args.lsi:
            (training_data, training_labels), (testing_data, testing_labels) = train_lsi_model(training, testing, args)
        else:
            (training_data, training_labels), (testing_data, testing_labels) = build_doc2vec_embeddings(training, testing, args)
        
        dist.update(testing_labels)
        
        for (clsName, clsModel) in getClassifiers(args):
            logging.info("Building %s classifier..." % clsName)
            clsModel.fit(training_data, training_labels)
            clsPreds = clsModel.predict(testing_data)
            
            f1Scores[ clsName ].append( metrics.f1_score(testing_labels, clsPreds) ) 
            accuracyScores[ clsName ].append( metrics.accuracy_score(testing_labels, clsPreds) ) 
    
    totalObs = float(sum(dist.values()))
    logging.info("Distribution: %s" % " | ".join([ "%s = %0.2f%%" % (k, 100*dist[k]/totalObs) for k in dist]))
    
    for clsName in accuracyScores:
        clsF1 = f1Scores[ clsName ]
        clsAccuracy = accuracyScores[ clsName ]
        
        logging.info( "%15s] F1: %.4f +/- %.4f  Accuracy: %.4f +/- %.4f" %
                        (clsName, np.mean(clsF1), np.std(clsF1),
                         np.mean(clsAccuracy), np.std(clsAccuracy),) )
        
def main():
    
    parser = argparse.ArgumentParser(description='Run Doc2Vec then push those vectors into Scikit-Learn')
    
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('-v', '--verbose', action="store_true", help="Show verbose output")
    
    parser.add_argument("-s", "--vecsize", type=int, default=100, help="Vector size")
    
    parser.add_argument("-k", default=5, type=int, help="K for cross-validation")
    parser.add_argument('--nostop', action="store_true", help='Test data path')
    parser.add_argument('--stem', action="store_true", help='Test data path')
    parser.add_argument("--dataset", default="sentiment140", help="Which dataset to use")
    
    parser.add_argument("--datapath", help="Path to chosen dataset, required for first use.")
    
    parser.add_argument("--dvSample", default=0.0001, type=float, help="Doc2Vec sampling.")
    parser.add_argument("--dvNegative", default=5, help="Doc2Vec negative.")
    parser.add_argument("--dvMinCount", default=1, help="Doc2Vec min_count.")
    parser.add_argument("--window", default=1, help="Doc2Vec window.")
    parser.add_argument("--dvWorkers", default=1, help="Doc2Vec workers.")
    
    parser.add_argument("--lsi", action="store_true", help="Use Latent Semantic Indexing.")
    
    parser.add_argument("--dataLength", default=None, type=int, help="Use to limit the number of examples used")
    parser.add_argument("--dataSample", default=None, type=float, help="Use to sample examples from data")
    
    
    parser.add_argument("--nTrees", default=15, type=int, help="Number of trees for Random Forests.")
    parser.add_argument("--rfFeatures", default="sqrt", choices=["sqrt","log2","auto","all"],
                        help="Number of features for Random Forests.")
    
    parser.add_argument("--learningRate", default=0.05, type=float,help="GloVe learning rate.")
    parser.add_argument("--verbose", action="store_true", help="Turn on verbose output.")
    
    parser.add_argument("--pca", action="store_true", help="Use pca with GloVe vectors")
    parser.add_argument('--parallelism', '-p', action='store',
                        default=4,
                        help=('Number of parallel threads to use'))

    parser.add_argument("--embeddings", choices=["glove","doc2vec"], default="doc2vec", help="Methods to generate vectors from text")
    
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)-15s: %(message)s', level=logging.INFO)

    all_data = list( loader.read( args.dataset,
                                 dataPath   = args.datapath,
                                 limit      = args.dataLength,
                                 sampleRate = args.dataSample ) )
    
    random.shuffle(all_data)
    test_model( all_data, args )
            
if __name__ == "__main__":
    main()
