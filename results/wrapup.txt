################################################################################
Executive Summary
################################################################################
Sunny-Side-Up: Summary of Results

Lab41's Sunny-Side-Up (SSU) Challenge benchmarked several Machine Learning (ML) methods for sentiment analysis.  The overall goal of the effort was to address the question:
    "Are often-hyped Deep Learning techniques worth pursuing for automatically classifying the sentiment of foreign Tweets?"

Overall assessment/recommendation: TODO

Lab41 worked over the course of several months with several developers, data scientists, and linguists to run several experiments that address various aspects of that overarching question:
    Performance of Deep Learning vs. Baseline ML
        - How well do Deep Learning techniques compare to more traditional ML methods?
        - How well do character-based Deep Learning architectures perform?
        - Which binary classifier works best for sentiment analysis?

    Deep Learning Environment

    Data Considerations
        - When using embedded word vectors, how much does the embedding model affect classification performance?
        - How important is the sentiment label to classification performance?
        - Is it possible to classify sentiment on minimally-processed foreign text?






################################################################################
Performance
################################################################################
Q: How well do Deep Learning architectures compare to baseline ML methods on microblogging messages?
A:

    Datasets (only using ones from microblogging that have good embedding models and differentiating labels):
        Sentiment140 (balanced set of 1.6M - 800K each positive and negative)
        Sina Weibo Censored (balanced set of TODO Romanized - TODO each positive and negative based on post CENSORED)

    Sentiment140: 200d-Glove-Twitter
        Crepe CNN
        Keras LSTM
        Gaussian
        LogisticRegression
        LinearSVC
        RandomForests

    Sina Weibo: 200d-Word2Vec-SinaWeibo
        Crepe CNN
        Keras LSTM
        Gaussian
        LogisticRegression
        LinearSVC
        RandomForests



Q: How well do character-based Deep Learning architectures perform?
A:

    Datasets (only using ones from microblogging that have good embedding models and differentiating labels):
        Sentiment140 (balanced set of 1.6M - 800K each positive and negative)
        Sina Weibo Censored (balanced set of TODO Romanized - TODO each positive and negative based on post CENSORED)

    Sentiment140: Char-encoding
        Crepe CNN
        Keras LSTM
        LogisticRegression (200d-Glove-Twitter)
        RandomForests (200d-Glove-Twitter)

    Sina Weibo: Char-encoding
        Crepe CNN
        Keras LSTM
        LogisticRegression (200d-Word2Vec-SinaWeibo)
        RandomForests (200d-Word2Vec-SinaWeibo)




Q: How much does classifier matter?
A: Assuming word embedding covers vocabulary, most ML classifiers do well with differentiable labels

    Datasets (only using ones with good embedding models and differentiating labels):
        IMDB (balanced set of 50K - 25K each positive and negative)
        Sentiment140 (balanced set of 1.6M - 800K each positive and negative)
        Sina Weibo Censored (balanced set of TODO Romanized - TODO each positive and negative based on post CENSORED)

    Test multiple classifiers:
        IMDB: 300d-Word2Vec-GoogleNews
            Gaussian
            LogisticRegression
            LinearSVC
            RandomForests

        Sentiment140: 200d-Glove-Twitter
            Gaussian
            LogisticRegression
            LinearSVC
            RandomForests

        Sina Weibo Censored: 200d-Word2Vec-SinaWeibo
            Gaussian
            LogisticRegression
            LinearSVC
            RandomForests



################################################################################
Environment
################################################################################
Summary graphic of pipeline: Data -> Model -> Classifier
Summary graphic of environment: Docker, IPython, GPUs, etc




################################################################################
Data
################################################################################
Q: How much does embedding model matter?
A: Size is not as relevant, but dictionary hits/misses matters a lot

    Datasets:
        IMDB (balanced set of 50K - 25K each positive and negative)
        Amazon (balanced set of 1.6M - 800K each positive and negative based on review over/under 3 stars)
        Sina Weibo (balanced set of 1.6M Romanized - 800K each positive and negative based on post DELETED)
        Sentiment140 (balanced set of 1.6M - 800K each positive and negative)

    Test pretrained:
        Word2vec pretrained on Google News (vocabulary size: 3,000,000):
            IMDB
            Amazon
            Sentiment140
            Sina Weibo

        GloVe pretrained on Tweets (vocabulary size: 1,193,514):
            IMDB
            Amazon
            Sentiment140
            Sina Weibo

    Observation: Poor performance on Amazon and Sina Weibo -- investigate whether performance due to mismatch between vocabulary in pretrained vectors vs. test dataset

    Test pretrained vs. dataset-specific vectors:
        Sentiment140
            word2vec pretrained on Google News  (vocabulary size: 3,000,000):
            GloVe pretrained on Tweets          (vocabulary size: 1,193,514):
            trained on 80% of dataset           (vocabulary size: 83,586):

        Sina Weibo
            word2vec pretrained on Google News  (vocabulary size: 3,000,000):
            GloVe pretrained on Tweets          (vocabulary size: 1,193,514):
            trained on 80% of dataset           (vocabulary size: 198,535):

        Amazon
            word2vec pretrained on Google News  (vocabulary size: 3,000,000):
            GloVe pretrained on Tweets          (vocabulary size: 1,193,514):
            word2vec trained on 80% of dataset  (vocabulary size: 143,386):

    Assessment: Comparable performance for Sentiment140-trained vectors suggests dataset-trained vectors can perform well enough; Amazon and Sina Weibo 50% performance suggests either: A) mismatch between vocabularies; B) sentiment labels on Amazon and Sina Weibo are bad




Q: How much does sentiment label matter?
A: Very important to have label that differentiates classes

    Datasets:
        Amazon (balanced set of 1.6M - 800K each positive and negative based on review over/under 3 stars)
        Sina Weibo (balanced set of 1.6M Romanized - 800K each positive and negative based on post DELETED)

    Test different labels on same dataset:
        Sina Weibo (balanced set of 1.6M Romanized - 800K each positive and negative based on post DELETED)
            word2vec pretrained on Google News (vocabulary size: 3,000,000):
            GloVe pretrained on Tweets (vocabulary size: 1,193,514):
            word2vec trained on 100% of dataset (vocabulary size: TODO):

        Sina Weibo (balanced set of TODO Romanized - TODO each positive and negative based on post CENSORED)
            word2vec trained on 100% of dataset (vocabulary size: TODO):
            word2vec pretrained on Google News (vocabulary size: 3,000,000):
            GloVe pretrained on Tweets (vocabulary size: 1,193,514):

    Assessment: Performance TODO for Sina Weibo's CENSORED vs. DELETED label suggests TODO



Q: Is it possible to classify sentiment on minimally-processed foreign text?
A:
    Datasets (using English and foreign microblogging)
        Sentiment140 (balanced set of 1.6M - 800K each positive and negative)
        Sina Weibo Censored (balanced set of TODO Romanized - TODO each positive and negative based on post CENSORED)

    Test on English, Foreign processed, and foreign non-processed
        Sentiment140: 200d-Glove-Twitter
            Gaussian
            LogisticRegression
            LinearSVC
            RandomForests

        Sina Weibo Censored: 200d-Word2Vec-SinaWeibo-Romanized
            Gaussian
            LogisticRegression
            LinearSVC
            RandomForests

        Sina Weibo Censored: 200d-Word2Vec-SinaWeibo-Hanzi
            Gaussian
            LogisticRegression
            LinearSVC
            RandomForests
