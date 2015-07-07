#!/bin/bash

INPUT_DIR="../input-test"
OUTPUT_DIR="../input-test"

echo "Converting PDFs from $INPUT_DIR to PNGs in $OUTPUT_DIR ..."

for i in `ls $INPUT_DIR/*.pdf`;
do
  FILENAME=`basename $i .pdf`;
  echo "Converting $i ..."
  convert -density 300 $INPUT_DIR/$FILENAME.pdf -quality 90 $INPUT_DIR/$FILENAME.png;
done

