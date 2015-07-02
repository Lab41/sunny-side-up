import sys
import getopt
from nltk import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.pipeline import Pipeline
from import_stanford_twitter import open_stanford_twitter_csv
from feature_evaluator import evaluate_features
from feature_extractors import word_feats
from ingest_twitter import split_tweets


def main(argv):
    # Initial local path for Stanford Twitter Data
    PATH = './StanfordTweetData/training.1600000.processed.noemoticon.csv'
    FEAT_PATH = './twitter_features.txt'

    # Parse command line arguments
    try:
        long_flags = ["help", "bernoulli", "multinomial", "gaussian"]
        opts, args = getopt.getopt(argv, "hi:f:", long_flags)
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    # Classifier variable. Used for training on tweet features below
    classifier = NaiveBayesClassifier
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()
        elif opt == '-i':
            # Updates PATH to Stanford Tweet CSV data set
            if arg:
                PATH = arg
            else:
                print('Argument expected for the -i option\n')
                usage()
                sys.exit(2)
        elif opt == '-f':
            FEAT_PATH = arg
        elif opt in ("bernoulli", "multinomial", "gaussian"):
            print("WARNING: Chosen classifier increases testing time")
            pipeline = None
            if opt == "bernoulli":
                pipeline = Pipeline([('nb', BernoulliNB())])
            elif opt == "multinomial":
                pipeline = Pipeline([('nb', MultinomialNB())])
            elif opt == "gaussian":
                pipeline = Pipeline([('nb', GaussianNB())])
            classifier = SklearnClassifier(pipeline)

    # Perform tweet parsing and learning
    print("Opening CSV file...")
    print("Extracting Features...")

    all_data = list()
    # Checks if all_data has already been set
    if any([opt == '-f' for opt, arg in opts]):
        tweet_feats = open(FEAT_PATH, 'r')
        all_data = [eval(line) for line in tweet_feats]
    else:
        all_data = open_stanford_twitter_csv(PATH, feat_extractor=word_feats)

    print("CSV file opened and features extracted")
    train_set, dev_set, test_set = split_tweets(all_data, train=.9,
                                                dev=0, test=.1, shuffle=True)
    print("Data split into sets")

    classifier.train(train_set)
    print("Classifier trained")

    print("Evaluating accuracy and other features\n")
    evaluate_features(classifier, test_set)


def usage():
    print('Usage: doall.py [-i file | -h]')
    print('Options and arguments:')
    print('-h\t\t: print this help message and exit (also --help)')
    print('-i csv_file\t: specify path for StanfordTweet input CSV file')
    print('-f txt_file\t: specify path for txt file containing tweet features')
    print('')
    print('--bernoulli\t: specifies the use a Bernoulli Naive Bayes classifier')
    print('\t\t  from the nltk\'s scikit learn integration')
    print('--multinomial\t: specifies the use a Multinomial Naive Bayes')
    print('\t\t  classifier from the nltk\'s scikit learn integration')
    print('--gaussian\t: specifies the use a Gaussian Naive Bayes classifier')
    print('\t\t  from the nltk\'s scikit learn integration')
    print('--help\t\t: print this help message and exit (also -h)\n')

if __name__ == '__main__':
    main(sys.argv[1:])
