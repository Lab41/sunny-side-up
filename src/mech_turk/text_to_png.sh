#!/bin/bash

# fail on errors
set -e

# specify dirs
__dir_input=$1
__dir_output=$2

# input verification
if [ "$__dir_input" == "" ] || [ "$__dir_output" == "" ]; then
  echo "You must provide input/output directories:"
  echo "$0 <input-directory-of-tweets> <output-directory-of-images>"
  exit 1
fi

# ensure output directory exists
mkdir --parents $__dir_output

# process files
cd $__dir_input
for filename in $(ls $__dir_input); do
  cat $filename | perl -n -mHTML::Entities -e ' ; print HTML::Entities::decode_entities($_) ;' > /tmp/in.txt && pango-view /tmp/in.txt --no-display --font "Scheherazade 24" -o $__dir_output/$filename.png
done
