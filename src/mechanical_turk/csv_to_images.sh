#!/bin/bash

# create images for each tweet
python csv_to_images.py /data/data/input/may_part-r-00001_ar /data/data/output/may_part-r-00001_ar 500

# migrate CSV of base64-encoded images
cp /data/data/output/may_part-r-00001_ar/may_part-r-00001_ar_output.csv /data
