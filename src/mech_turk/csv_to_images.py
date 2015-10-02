# coding=utf8
import sys, os
import csv, codecs, base64, HTMLParser
from subprocess import call


def text_to_png(dir_input, dir_output, tempfile = '/tmp/tweet.txt', debug=False):
    '''

      create png images from each text file in directory

      input:  directory of text files
      output: png renderings of text files

    '''

    try:

        # unescape html characters
        html_parser = HTMLParser.HTMLParser()

        # process all files in directory
        for root, dirs, files in os.walk(dir_input):

            # process everything at once
            dirs.extend(files)

            # iterate files
            for pathname in dirs:
                file_in = os.path.join(root, pathname)
                file_out = "{}.png".format(os.path.join(dir_output, pathname))

                # optional feedback
                if (debug):
                    print("processing file {}".format(file_in))

                # save the processed text to a temp file
                with codecs.open(file_in, 'r', encoding='utf-8') as fh:

                    # decode HTML entities
                    data = html_parser.unescape( fh.read() )

                    # save the temp file
                    with codecs.open(tempfile, 'w', encoding='utf-8') as outfile:
                        outfile.write(data)

                # render the text as an image
                call(['pango-view',tempfile,'--no-display','--font','Scheherazade 24','-o',file_out])


    # catch-all for exceptions
    except Exception as e:
        print(e)


def png_to_csv(dir_input, filename='output.csv', csv_output_head = ['batch_id', 'tweet1', 'tweet2', 'tweet3', 'tweet4', 'tweet5'], debug=False):
    '''

      base64-encode all png files in directory and save to single csv file

      input:  directory of png images
      output: csv file with base64-encoded images in each cell

    '''

    # store csv as list of lists
    data = []
    row = []
    images_per_row = len(csv_output_head)

    # iterate and count number of
    counter = 0

    # process all files
    for root, dirs, files in os.walk(dir_input):

        # process everything at once
        dirs.extend(files)

        # iterate files
        for pathname in dirs:
            file_in = os.path.join(root, pathname)

            # optional feedback
            if (debug):
                print("processing file {}".format(file_in))

            # add base64-encoded image to CSV row
            encoded = base64.b64encode(open(file_in, "rb").read())
            row.append(encoded)

            # create list-of-lists structure for csv
            if (counter % images_per_row == 0):

                # add id to row
                row.insert(0, counter / images_per_row)

                # add row to CSV and re-initialize row
                data.append(row)
                row = []

            # track number processed (for building rows)
            counter += 1


    # add the last row
    if (len(row) > 0):
        data.append(row)


    # write CSV
    with open(os.path.join(dir_input, filename), "wb") as fh:
        writer = csv.writer(fh)

        # write header
        writer.writerow(csv_output_head)

        # write rows
        writer.writerows(data)



if __name__ == "__main__":

    # ensure params specified
    if (len(sys.argv) < 3):
      msg = """
        Insufficient arguments!  Usage: \n\
          {} <input-directory> <output-directory>
        """.format(sys.argv[0])
      raise Exception(msg)

    # setup input/output directories
    dir_input = sys.argv[1]
    dir_output = sys.argv[2]

    # ensure input directory exists
    if not os.path.exists(dir_input):
        raise Exception("Bad directory: {}".format(dir_input))

    # ensure output directory exists
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # create images for each tweet
    #text_to_png(dir_input, dir_output)

    # store images into CSV for mechanical turk
    png_to_csv(dir_output)
