#!/bin/bash
#
#    mover   -    separates the rgb and depth images into two separate datasets. The destination folders need to exist
#                 already
#
#    usage:    mover
#
#  David González León, Jade Gröli    2021-11-19
dir="double_image_dataset"
cd $dir
for i in $(seq 0 1 9); do
   cd $i
   echo "Copying for $i..."
   for ext in $(ls *.png); do
      extracted="${ext%.*}"
      extracted="${extracted#*_}"
      dest_dir=""
      if [[ $extracted = "Color" ]];then
         dest_dir="rgb_image_dataset"
      else
         dest_dir="depth_image_dataset"
      fi
      dest_dir="../../$dest_dir/$i/$ext"
      cp $ext $dest_dir
   done
done