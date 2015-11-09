all:
	python setup.py build_ext

get-wiki:
	wget http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-meta-current1.xml-p000000010p000010000.bz2

process-wiki:
	python -- examples/example.py -w -c enwiki-latest-pages-meta-current1.xml-p000000010p000010000.bz2

train-wiki:
	python -i -- examples/example.py -t 30 -p 2

all-wiki: get-wiki process-wiki train-wiki

glove-get-analogies:
	wget --output-document=./data/questions-words.txt https://word2vec.googlecode.com/svn/trunk/questions-words.txt

glove-build-stanford:
	python data/models-load-stanford.py -o /data/src/glove/data

glove-evaluate-analogies:
	python -m examples/analogy_tasks_evaluation -t ./data/questions-words.txt -m glove.6B.50d.obj

glove-evaluate-similarities:
	python examples/models-test-similarity.py -m data/glove.6B/glove.6B.50d.obj

glove-save-and-load:
	python examples/save_and_load.py

all-tests:
	for t in $$(ls glove/tests/*.py); do python -m nose $$t; done

all-analogies: glove-get-analogies glove-build-stanford glove-evaluate-analogies

all-similarities: glove-evaluate-similarities glove-save-and-load

.PHONY: all get-wiki process-wiki train-wiki glove-get-analogies glove-build-stanford glove-evaluate-analogies glove-evaluate-similarities glove-save-and-load
