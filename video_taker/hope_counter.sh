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
echo "$A/8'000 sequences"
A=$A/80
echo "$A% du dataset construit"
