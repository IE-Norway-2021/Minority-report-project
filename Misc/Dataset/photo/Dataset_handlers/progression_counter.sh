#!/bin/bash
#
#    progression_counter    -    counts the progress of the dataset over a fixed goal (in our case 24'000 pictures)
#
#    usage:    progression_counter
#
#  David González León, Jade Gröli    2021-11-19
declare -i A
for ext in *; do
if [[ -d $ext ]];then
cd $ext
A=$A+$(ls | wc -l)
cd ..
fi
done
echo "$A/24'000 photos"
A=$A/240
echo "$A% du dataset construit"