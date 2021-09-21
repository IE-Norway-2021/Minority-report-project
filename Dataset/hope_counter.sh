declare -i A
for ext in *; do
if [[ -d $ext ]];then
cd $ext
A=$A+$(ls | wc -l)
cd ..
fi
done
echo "$A/10'000 photos"
A=$A/100
echo "$A% du dataset construit"