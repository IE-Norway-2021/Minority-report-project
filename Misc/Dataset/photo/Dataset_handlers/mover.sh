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