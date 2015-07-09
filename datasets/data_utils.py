import os, io
import csv
import logging
from progressbar import ProgressBar
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
        fname = './.downloads/' + url.split('/')[-1]
        if fname in os.listdir('.'):
            print("File has already been dowloaded")
            return fname

        # Create hidden folder to hold zip file
        os.mkdir(os.path.join(os.getcwd(), '.downloads')

        response = urlopen(url)
        # Get total length of content
        total_size = int(response.info().getheader('Content-Length').strip())
        chunk_size = total_size / 100

        # Open dowload file and save locally
        with open(fname, 'wb') as f:
            # Initialize the visual progress bar for the download
            pbar = ProgressBar().start()
            for i in range(100):
                f.write(response.read(chunk_size))
                pbar.update(i+1)
            pbar.finish()
        return fname

    # Handle errors
    except HTTPError, e:
        print "HTTP Error:", e.code, url
    except URLError, e:
        print "URL Error:", e.reason, url
