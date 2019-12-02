#!/usr/bin/env bash

END=345

# Expand files into a temporary directory
mkdir ./jobs
for i in $(seq -f "%04g" 100  $END)
do
  cat job-template.yaml | sed "s/\$ITEM/$i/" > ./jobs/job-$i.yaml
done