import os, sys, random
import re
import csv
from src.datasets.data_utils import mkdir_p

# update field size
csv.field_size_limit(sys.maxsize)

# ingest file path
file_path_in = '/data/openweibo/'
file_path_out = '/data/openweibocensored'

# randomly keep some negative samples
randomprob = 2000
randchoice = int(random.random()*randomprob)

# track numbers
num_positive = 0
num_negative = 0

# get list of weekNN.csv files at file_path
ow_files = [ os.path.join(file_path_in, f) for f in os.listdir(file_path_in) if re.match(r"week[0-9]{,2}\.csv", f) is not None ]
ow_files.sort()

# ensure directory exists
if not os.isdir(file_path_out):
    mkdir_p(file_path_out)

# create csv file
with open(os.path.join(file_path_out, 'censored.csv'), 'wb') as outfile:

    # object to write csv file
    csv_writer = csv.writer(outfile, delimiter=',')

    # search all files
    for table_path in ow_files:
        with open(table_path, 'rbU') as f:
            print("checking in file {}".format(table_path))

            # save line if post was censored
            for line in csv.reader(f, dialect=csv.excel):
                if len(line) > 10:
                    if line[10] is not '':
                        csv_writer.writerow(line)

                        # track how many positives kept
                        num_positive += 1
                        if num_positive % 100 == 0:
                            print("Added positive: {}".format(num_positive))

                    else:

                        # randomly keep some negative samples
                        if int(random.random()*randomprob)==randchoice:
                            csv_writer.writerow(line)

                            # track how many negatives kept
                            num_negative += 1
                            if num_negative % 100 == 0:
                                print("Added negative: {}".format(num_negative))
