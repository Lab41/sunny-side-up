import sys
import getopt
from nltk import NaiveBayesClassifier
from import_stanford_twitter import open_stanford_twitter_csv
from feature_evaluator import evaluate_features
from feature_extractors import word_feats
from ingest_twitter import split_tweets


def main(argv):
    # Initial local path for Stanford Twitter Data
    PATH = './StanfordTweetData/training.1600000.processed.noemoticon.csv'

    # Parse command line arguments
    try:
        opts, args = getopt.getopt(argv, "hi", ["help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

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

    # Perform tweet parsing and learning
    print("Opening CSV file...")
    print("Extracting Features...")
    all_data = open_stanford_twitter_csv(PATH, feat_extractor=word_feats)
    print("CSV file opened and features extracted")
    train_set, dev_set, test_set = split_tweets(all_data, train=.9,
                                                dev=0, test=.1, shuffle=True)
    print("Data split into sets")
    classifier = NaiveBayesClassifier.train(train_set)
    print("Classifier trained")
    print("Evaluating accuracy and other features")
    evaluate_features(classifier, test_set)


def usage():
    print('Usage: doall.py [-i file | -h]')
    print('Options and arguments:')
    print('-h\t: print this help message and exit (also --help)')
    print('-i file\t: specify path for StanfordTweet input CSV file\n')
    print('--help\t: print this help message and exit (also -h)\n')

if __name__ == '__main__':
    main(sys.argv[1:])
