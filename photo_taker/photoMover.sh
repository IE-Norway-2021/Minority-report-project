#!/bin/bash

cd cmake-build-debug/Images
for ext in *; do
  cp -R $ext ~/Documents/Minority-report-project/Dataset
done



