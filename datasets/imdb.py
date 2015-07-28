import tarfile
import random
from data_utils import get_file


# Dictionary that defines the Sentiment features
Sentiment = {0: 'neg', 2: 'neutral', 4: 'pos'}


def load_data(file_path=None):
    ''' Function that takes in a path to the IMDB movie review dataset
        word analogy file, opens it, removes topic tags and returns a list
        of the analogies

        @Arguments:
            file_path -- (optional) personal system file path to the
                IMDB data set in gzip form(or others of
                a similar structure)

        @Return:
            A generator over a tuples of Movie reviews and their sentiment
    '''
    tar = None

    # Open file path
    if not file_path:
        file_path = get_file("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")

    tar = tarfile.open(file_path, mode="r:gz")
    # Specifies positive and negative files
    pos = [(m, Sentiment[4]) for m in tar.getmembers() if r'pos/' in m.name]
    neg = [(m, Sentiment[0]) for m in tar.getmembers() if r'neg/' in m.name]

    # Combines data and shuffles it.
    all_data = pos + neg
    random.shuffle(all_data)

    for (member, sentiment) in all_data:
        # Open the movie review
        f = tar.extractfile(member)
        yield (f.read(), sentiment)
        # Closes f on the following next() call by user
        f.close()
    tar.close()
