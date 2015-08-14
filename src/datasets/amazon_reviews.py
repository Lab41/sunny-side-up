import gzip
import json
from data_utils import get_file


def load_data(file_path=None, verbose=False):
    ''' Function that takes in a path to the Stanford SNAP Amazon review
        data, opens it, and yields a dictoray of information for each
        review

        @Arguments:
            file_path -- (optional) personal system file path to the
                SNAP Stanford data set (or others of a similar structure)

        @Return:
            A generator over a dictionaries of each Amazon Reveiws
    '''
    # Open file path
    if not file_path:
        file_path = get_file("https://snap.stanford.edu/data/amazon/all.txt.gz")

    # Parse Amazon Reviews GZip file -- taken from Stanford SNAP page
    try:
        f = gzip.open(file_path, 'r')
    except IOError, e:
        print "IO Error", e.code, file_path

    entry = dict()
    for l in f:
        l = l.strip()
        colonPos = l.find(':')
        if colonPos == -1:
            # JSON.loads(JSON.dumps) converts JSON to
            # Python dictionary type
            yield json.loads(json.dumps(entry))
            entry = {}
            continue
        eName = l[:colonPos]
        rest = l[colonPos+2:]
        entry[eName] = rest
    # Returns Generator object
    yield json.loads(json.dumps(entry))
