#!/bin/bash
#
#    mkdir    -   makes the directory used for containing the dataset. Makes all the subfolders
#
#    usage:    mkdir
#
#  David González León, Jade Gröli    2021-11-19
dirname="depth_200x120_image_dataset"
mkdir $dirname
cd $dirname
for i in $(seq 0 1 9); do
   mkdir $i
done
