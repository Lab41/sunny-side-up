import csv

# Dictionary that defines the Sentiment features
Sentiment = {0: 'neg', 2: 'neutral', 4: 'pos'}

def latin_csv_reader(csv_data, dialect=csv.excel, **kwargs):
    ''' Function that takes an opened CSV file with
        ASCII or UTF-8 encoding and convert's to Latin-1

        @Arguments:
            csv_data -- a CSV file opened for reading

            dialect -- specifies the file dialect type

            **kwargs -- other typical arguments one would pass
                into the csv.reader() function. 
                Ex: delimiter=','

        @Return:
            A python generator over the lines in the given
            CSV file
    '''

    # Opens CSV reader
    csv_reader = csv.reader(csv_data, dialect=dialect, **kwargs)
    # Converts row to latin-1 encoding and yields this line on next() calls
    for row in csv_reader:
        yield [unicode(cell, 'latin-1') for cell in row]


def open_stanford_twitter_csv(full_file_path, feat_extractor=None):
    ''' Function that takes in a path to the StanfordTweetData CSV
        file, opens it, and adds tweet strings and their respective
        sentiments to a list

        @Arguments:
            full_file_path -- full system file path to the 
                training.1600000.processed.noemoticon.csv data set (or others of
                a similar structure)

                The Stanford Data set if of the following format per row:
                    [polarity, tweet id, tweet date, query, user, tweet text]

            feat_extractor -- a function that converts a tweet text string
                and outputs a dictionary of features

        @Return:
            A list of tuples of the following format:
                (tweets/features, sentiment label)
    '''

    tweet_to_sentiment = list()
    with open(full_file_path, 'r') as twitter_csv:
        # Open CSV file for reading in 'latin-1'
        reader = latin_csv_reader(twitter_csv, delimiter=',')
        for tweet in reader:
            # Gets tweets string from line in csv
            tweet_string = tweet[5]
            # Gets feature from Sentiment dictionary
            sent = Sentiment(int(tweet[0]))
            # If a feat_extractor function was provided, apply it to tweet
            if feat_extractor:
                features = feat_extractor(tweet_string)
                tweet_to_sentiment.append(features, sent)
            else:
                tweet_to_sentiment.append((tweet_string, sent))
    return tweet_to_sentiment

def tweets_to_txt(read_path, write_path):
    ''' Function that takes in a path to the StanfordTweetData CSV
        file, opens it, and writes the tweets with new lines to an output
        file.
    '''

    with open(read_path, 'r') as twitter_csv, open(write_path, 'w') as output:
        reader = latin_csv_reader(twitter_csv, delimiter=',')
        # For each line in CSV, write each tweet with a new line to the output
        for line in reader:
        	output.write(line[5].encode('UTF-8') + '\n')

