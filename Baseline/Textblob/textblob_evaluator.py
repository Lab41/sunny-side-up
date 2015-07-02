import collections
import nltk


def textblob_evaluator(textblob_data):
    ''' Takes in an previously trained_classifier and test_data
        with which to determine the classifiers accuracy.

        trained_classifier:
                Type: nltk.classify.api.ClassifierI

        test_data:
                Of the following form:
                [
                        (feature set 1, label 1)
                        (feature set 2, label 2)
                        ....
                        (feature set n, label n)
                ]
    '''

    # Data pre-processing for analysis
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)

    # Dictionary that defines the Sentiment features
    Sentiment = {0: 'neg', 2: 'neutral', 4: 'pos'}
    pos = Sentiment[4]
    neg = Sentiment[0]
    correct_count = 0

    print("Classifying test data...")
    for i, (txtblb, label) in enumerate(textblob_data):
        referenceSets[label].add(i)
        if txtblb.sentiment[0] > 0:
            testSets[pos].add(i)
            if label == pos:
                correct_count += 1
        else:
            testSets[neg].add(i)
            if label == neg:
                correct_count += 1

    # Prints metrics to show how well the Textblob Sentiment performs
    print 'Test on {} instances'.format(len(textblob_data))
    print '--------------------------------'
    print 'accuracy:', correct_count / float(len(textblob_data))
    print 'pos precision:', nltk.metrics.precision(referenceSets[pos],
                                                   testSets[pos])
    print 'pos recall:', nltk.metrics.recall(referenceSets[pos],
                                             testSets[pos])
    print 'neg precision:', nltk.metrics.precision(referenceSets[neg],
                                                   testSets[neg])
    print 'neg recall:', nltk.metrics.recall(referenceSets[neg],
                                             testSets[neg])
