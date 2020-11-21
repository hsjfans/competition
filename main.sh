#! /bin/bash 
echo 'Start --------------------------------'

createIfNotExists(){
  file_path=$1
  if [ ! -d ${file_path} ];then
  mkdir ${file_path}
    echo "${file_path} build success ---"
  else 
    echo "${file_path} has exists ----"
  fi
}

file_paths=('./features/','./feature_importance','./prediction','./submit')
for file_path in file_paths
do
   createIfNotExists(${file_path})
done

cd ./feature_engineering
echo 'feature engineering .......'
./feature_engineering.sh 
echo 'feature engineering end .......'
cd ../feature_select
echo 'feature select .......'
python model.py
echo 'feature select end .......'
cd ..

echo 'Finish --------------------------------'
