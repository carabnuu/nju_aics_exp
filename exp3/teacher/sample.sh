#!/bin/bash


echo "------------------------------"
python3.7 ./process.py
echo "Image processed successfully"
echo "------------------------------"

export APP_SOURCE_PATH=$(pwd)
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub

model_name="MyFirstApp"

cd ${APP_SOURCE_PATH}/data

if [ -d ${APP_SOURCE_PATH}/build/intermediates/host ];then
	rm -rf ${APP_SOURCE_PATH}/build/intermediates/host
fi

mkdir -p ${APP_SOURCE_PATH}/build/intermediates/host
cd ${APP_SOURCE_PATH}/build/intermediates/host

cmake ../../../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE

make

if [ $? == 0 ];then
	echo "make for app ${model_name} Successfully"
else
	echo "make for app ${model_name} failed"
fi

cd ${APP_SOURCE_PATH}
echo "------------------------------"
echo "Start inference"
./main
echo "------------------------------"
