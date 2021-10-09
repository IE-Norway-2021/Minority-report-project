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