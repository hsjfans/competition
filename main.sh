#! /bin/bash 
echo 'Start --------------------------------'
cd ./feature_engineering
./feature_engineering.sh &
cd ../feature_select
python model.py >& out
cd ..

echo 'Finish --------------------------------'
