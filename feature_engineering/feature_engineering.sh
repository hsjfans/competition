#! /bin/bash 

if [ ! -d "./features/" ];then
  mkdir features
  echo "features 文件夹已经 Create----"
else
  echo "features 文件夹已经存在----"
fi

runipy baseline.ipynb 
runipy annual_report_info.ipynb
runipy tax_info.ipynb
runipy news_info.ipynb
runipy other_info.ipynb
runipy change_info.ipynb
runipy feature.ipynb

echo "Feature Engineering Finish -------------------------------- \n"