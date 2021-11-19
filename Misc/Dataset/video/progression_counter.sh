#!/bin/bash
#
#    progression_counter    -    counts the progress of the dataset over a fixed goal (in our case 6'000 sequences
#                                of depth and rgb videos)
#
#    usage:    progression_counter
#
#  David González León, Jade Gröli    2021-11-19
declare -i A
cd Dataset
cd depth
for ext in *; do
  if [[ -d $ext ]];then
    cd $ext
    A=$A+$(ls | wc -l)
    cd ..
  fi
done
cd ..
echo "$A/6'000 sequences"
A=$A/60
echo "$A% du dataset construit"
cd ..