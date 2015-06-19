import random
import copy

def split_tweets(data, train=.7, dev=.2, test=.1, shuffle=False):
	''' Takes mapping of tweet to label and splits data into train, dev, and test 
		sets according to proportion given

		@Arguments:
			data -- The mapping of tweet string to sentiment. Should be of form
				[("tweet_string_1", label_1), ("tweet_string_2", label_2), ...]

			train (optional) -- Proportion of tweets to give the training set

			dev (optional) -- Proportion of tweets to give the development set

			test (optional) -- Proportion of tweets to give the testsing set

			shuffle (optional) -- Boolean value that if True randomly puts the 
				data into the various sets, rather than in the order of the file

		@Raises:
			ValueError -- If train + dev + test is not equal to 1

		@Return:
			Tuple containing the three sets of data:
				(train_set, dev_set, test_set)
	'''

	# Deals with issues in Floating point arithmetic
	if train * 10 + dev * 10 + test * 10 != 10:
		raise ValueError("Given set proportions do not add up to 1")


	if shuffle:
		# Deep copy handles case that data contains needed objects
		# data = copy.deepcopy(data)
		random.shuffle(data)

	data_size = len(data)
	train_size = int(train * data_size)
	dev_size = int(dev * data_size)

	# Partition data
	train_set = data[0:train_size]
	dev_set = data[train_size+1:train_size+dev_size]
	test_set = data[train_size+dev_size+1:data_size]
	
	return train_set, dev_set, test_set




