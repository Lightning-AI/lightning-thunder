#!/bin/bash

# Get the name of the file to process
filename=$1

# Create a temporary file to store the filtered lines
tempfile=$(mktemp)

# Loop through the original file and remove lines including the word "torch"
while read -r line; do
  if [[ ! "$line" =~ "torch" ]]; then
    echo "$line" >> "$tempfile"
  fi
done < "$filename"

# Move the temporary file to the original file
mv "$tempfile" "$filename"
