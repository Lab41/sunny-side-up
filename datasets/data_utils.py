import os, io
import csv
import logging
from urllib2 import urlopen, HTTPError, URLError


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


def get_file(url):
    try:
        # Prevents redownloading
        fname = os.path.join(os.getcwd(), '.downloads', url.split('/')[-1])
        if '.downloads' in os.listdir('.'):
            if url.split('/')[-1] in os.listdir('./.downloads'):
                print("File has already been dowloaded")
                return fname

        # Create hidden folder to hold zip file
        if not os.path.exists(os.path.join(os.getcwd(), '.downloads')):
            os.mkdir(os.path.join(os.getcwd(), '.downloads'))

        response = urlopen(url)

        # Open dowload file and save locally
        with open(fname, 'wb') as f:
            print("Downloading %s... " % url.split('/')[-1]),
            f.write(response.read())
            print("Success!")
        return fname

    # Handle errors
    except HTTPError, e:
        print "HTTP Error:", e.code, url
    except URLError, e:
        print "URL Error:", e.reason, url
