export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest


FILE=./atc
if [ ! -f "$FILE" ]; then
    ln -s /usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc ./
fi

# fp16量化
./atc --framework=5 --model="./model/vgg16.onnx" \
      --output="model/vgg16" --input_format=NCHW \
      --input_shape="image:1,3,224,224" \
      --log=debug \
      --soc_version=Ascend310 \
      --precision_mode=force_fp16

# 正常导出
./atc --framework=5 --model="./model/vgg16.onnx" \
      --output="model/vgg16_fp32" --input_format=NCHW \
      --input_shape="image:1,3,224,224" \
      --log=debug \
      --soc_version=Ascend310 \
      --precision_mode=force_fp32