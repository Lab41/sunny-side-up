import logging
from data_utils import get_file


def load_data(file_path=None):
    ''' Function that takes in a path to the Google questions-words.txt
        word analogy file, opens it, removes topic tags and returns a list
        of the analogies

        @Arguments:
            file_path -- (optional) personal system file path to the
                questions-words.txt data set (or others of
                a similar structure)

                The Questions-Words Dataset is of the following format per row:
                    'WordA WordB WordC, WordD'

        @Return:
            A list of strings representing analogies
    '''
    # TODO #
    # Add logging for a verbosity flag...although the file is small #
    # so I am not sure that it is even worth it #

    word_analogies = list()

    # Open file path
    if not file_path:
        file_path = get_file("https://word2vec.googlecode.com/svn/trunk/questions-words.txt")

    # Questions word file
    qw = open(file_path, 'r')

    ### Removes categories!!! ### 
    word_analogies = [l for l in qw.read().splitlines() if l[0] != ':']
    qw.close()
    return word_analogies
