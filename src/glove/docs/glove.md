## GloVe
Core module updates, data loaders, and test examples for the [glove-python](https://github.com/maciejkula/glove-python) module.

##### src/glove/Makefile
Makefile wrapper for various data processing commands (i.e. 'make all' wraps around 'python setup.py build_ext')


### Data Loaders
##### src/loader.py
Data loading generators for any of:
1. sentiment140 (sentiment140.csv)
2. amazon reviews (aggressive_dedup.json)
3. imdb reviews (aclImdb_v1.tar)

##### src/preprocess.py
Tweet tokenizer that splits on whitespace and substitutes '<url>' for all inline URLs

### Examples

##### src/glove/examples/analogy_tasks_evaluation.py
Evaluate word analogy performance by reading in word analogies, calculating analogous words from a trained GloVe model, and ranking the output relative to the expected/true analogy. Requires use of
```src/glove/glove/metrics/accuracy.py```
to parse and score word analogies of the form 'W1_from W1_to W2_from W2_to'.

##### src/glove/examples/example.py
Train a GloVe vector model based on a Wikipedia dump and query the model for words similar to an input word.

##### src/glove/examples/save_and_load.py
Train a GloVe vector model based on a specified input file, save the model object, and output a list of example word similarities.

##### src/sklearn_embeddings.py
Trains a GloVe or Doc2Vec model on the sentiment140 dataset and calculates train/test accuracy using Logistic Regression, Random Forests, and Gaussian Naive Bayes classifiers.
