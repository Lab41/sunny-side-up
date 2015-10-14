import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
import numpy as np

import amazon_reviews
import data_utils


def batch_data(data_loader, batch_size=128, normalizer_fun=data_utils.normalize, 
               transformer_fun=data_utils.to_one_hot, flatten=True):
    '''
    Batches data, doing all necessary preprocessing and
    normalization.
    '''

    docs = []
    labels = []

    for doc_text, label in data_loader:
        try:
            doc_text = normalizer_fun(doc_text)
            # transform document into a numpy array
            transformed_doc = transformer_fun(doc_text)
            docs.append(transformed_doc)
            labels.append(label)
        except data_utils.DataException as e:
            logger.info(e)
            continue
        if len(docs) >= batch_size:
            docs_np = np.array(docs)
            if flatten==True:
                # transform to form (batch_size, w*h); flattening doc
                docs_np = docs_np.reshape(batch_size,-1)
            # labels come out in a separate (batch_size, 1) np array
            labels_np = np.array(labels).reshape(batch_size, -1)
            docs = []
            labels = []

            yield docs_np, labels_np
