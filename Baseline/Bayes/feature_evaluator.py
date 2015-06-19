import collections
import nltk
from import_stanford_twitter import Sentiment


def evaluate_features(trained_classifier, test_data):
    ''' Takes in an previously trained_classifier and test_data
        with which to determine the classifiers accuracy.

        trained_classifier:
                Type: nltk.classify.api.ClassifierI

        test_data:
                Of the following form:
                {
                        (feature set 1, label 1)
                        (feature set 2, label 2)
                        ....
                        (feature set n, label n)
                }
    '''

    # Data pre-processing for analysis
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)
    pos = Sentiment[4]
    neg = Sentiment[0]

    for i, (features, label) in enumerate(test_data):
        referenceSets[label].add(i)
        predicted = trained_classifier.classify(features)
        testSets[predicted].add(i)

    # Prints metrics to show how well the feature selection did
    print 'Test on {} instances'.format(len(test_data))
    print 'accuracy:', nltk.classify.util.accuracy(trained_classifier,
                                                   test_data)
    print 'pos precision:', nltk.metrics.precision(referenceSets[pos],
                                                   testSets[pos])
    print 'pos recall:', nltk.metrics.recall(referenceSets[pos],
                                             testSets[pos])
    print 'neg precision:', nltk.metrics.precision(referenceSets[neg],
                                                   testSets[neg])
    print 'neg recall:', nltk.metrics.recall(referenceSets[neg],
                                             testSets[neg])
    trained_classifier.show_most_informative_features(20)
