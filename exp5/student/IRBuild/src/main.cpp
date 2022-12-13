
/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string.h>
#include "tensorflow_parser.h"
#include "caffe_parser.h"
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "attr_value.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_ir_build.h"
#include "all_ops.h"
//#include "softmax.h"
#include <dlfcn.h>
#include <unistd.h>

using namespace std;
using namespace ge;
using ge::Operator;

namespace {
static const int kArgsNum = 3; //输入参数个数
static const int kSocVersion = 1; //芯片型号
static const int kGenGraphOpt = 2;
static const std::string kPath = "../ckpt/"; //权重文件所属文件夹
}// namespace
 
/*
>>读取bin文件构造tensor，并赋给常量算子Const
    path：指定权重文件路径
    weight：从权重文件中读取的Tensor类型的权重数据
    len：指定权重数据大小
*/
bool GetConstTensorFromBin(string path, Tensor &weight, uint32_t len) {

    ifstream in_file(path.c_str(), std::ios::in | std::ios::binary);
    if (!in_file.is_open()) {
        std::cout << "failed to open" << path.c_str() << '\n';
        return false;
    }
    in_file.seekg(0, ios_base::end);
    istream::pos_type file_size = in_file.tellg();
    in_file.seekg(0, ios_base::beg);

    if (len != file_size) {
        cout << "Invalid Param.len:" << len << " is not equal with binary size (" << file_size << ")\n";
        in_file.close();
        return false;
    }
    char* pdata = new(std::nothrow) char[len];
    if (pdata == nullptr) {
        cout << "Invalid Param.len:" << len << " is not equal with binary size (" << file_size << ")\n";
        in_file.close();
        return false;
    }
    in_file.read(reinterpret_cast<char*>(pdata), len);
    auto status = weight.SetData(reinterpret_cast<uint8_t*>(pdata), len);
    if (status != ge::GRAPH_SUCCESS) {
        cout << "Set Tensor Data Failed"<< "\n";
        delete [] pdata;
        in_file.close();
        return false;
    }
    in_file.close();
    return true;
}


/*
>> 生成Conv2D算子：
    weight_shape:  用于定义卷积算子filter的权重shape
    conv_name：卷积算子的名称
    data：上一个算子。进行构图的连接
*/
Operator GenConv2dOp(Shape weight_shape,string conv_name,Operator data){

    //构造权重算子的描述信息desc_weight
    TensorDesc desc_weight(weight_shape,  FORMAT_NCHW, DT_FLOAT);
    //构造tensor 
    Tensor weight_tensor(desc_weight);
    //计算出tensor需要的大小
    uint32_t weight_len = weight_shape.GetShapeSize() * sizeof(float);
    //从bin文件中加载数据，赋给tensor
    bool res = GetConstTensorFromBin(kPath+conv_name+".weight.bin", weight_tensor, weight_len);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
    }
    //创建Const类型的权重算子，通过Const算子的属性value，传入tensor
    auto conv_weight = op::Const(conv_name+"_weight")
        .set_attr_value(weight_tensor);

    //创建卷积算子
    auto conv2d = op::Conv2D(conv_name)
        .set_input_x(data)    //定义输入，传入上一个算子
        .set_input_filter(conv_weight)    //定义卷积核，传入卷积核的权重
        .set_attr_strides({ 1, 1, 1, 1 })  //定义strides
        .set_attr_pads({ 1, 1, 1, 1 })   //定义pads
        .set_attr_dilations({ 1, 1, 1, 1 })  //定义dilations
        .set_attr_data_format("NCHW");  //定义输入数据的格式

    TensorDesc conv2d_input_desc_x(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
    TensorDesc conv2d_input_desc_filter(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
    TensorDesc conv2d_output_desc_y(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
    conv2d.update_input_desc_x(conv2d_input_desc_x);     //更新卷积的输入信息
    conv2d.update_input_desc_filter(conv2d_input_desc_filter);  //更新卷积的filter信息
    conv2d.update_output_desc_y(conv2d_output_desc_y);    //更新卷积的输出信息

    return conv2d;
}

/*
>> 生成Batch Normalization算子
    weight_shape：用于定义BN算子的四个输入权重的shape
    bn_name：BN算子的名称
    data：上一个算子。进行构图的连接
*/
Operator GenBNOp(Shape weight_shape,string bn_name, Operator data){
    TensorDesc desc_weight_1(weight_shape, FORMAT_ND, DT_FLOAT);
    //定义BN算子的四个权重Const算子，分别对应为BN的offset，scale，mean和variance
    Tensor offset_weight_tensor(desc_weight_1);
    Tensor scale_weight_tensor(desc_weight_1);
    Tensor mean_weight_tensor(desc_weight_1);
    Tensor variance_weight_tensor(desc_weight_1);

    uint32_t weight_1_len = weight_shape.GetShapeSize() * sizeof(float);
    //从bin文件中加载BN的offset，offset对应权重文件中的beta，表示输入偏置项
    bool res = GetConstTensorFromBin(kPath+bn_name+".beta.bin", offset_weight_tensor, weight_1_len);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
    }
    //从bin文件中加载BN的scale，scale对应权重文件中的gamma，表示输入Scalar
    res = GetConstTensorFromBin(kPath+bn_name+".gamma.bin", scale_weight_tensor, weight_1_len);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
    }
     //从bin文件中加载BN的moving_mean，表输入的均值   
    res = GetConstTensorFromBin(kPath+bn_name+".moving_mean.bin", mean_weight_tensor, weight_1_len);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
    }
    //从bin文件中加载BN的moving_variance，表输入的方差
    res = GetConstTensorFromBin(kPath+bn_name+".moving_variance.bin", variance_weight_tensor, weight_1_len);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
    }   

    //构造对应的常量算子，用来定义权重
    auto bn_offset = op::Const(bn_name+"_beta")
        .set_attr_value(offset_weight_tensor);
    auto bn_scale = op::Const(bn_name+"_gamma")
        .set_attr_value(scale_weight_tensor);
    auto bn_mean = op::Const(bn_name+"_mean")
        .set_attr_value(mean_weight_tensor);
    auto bn_variance = op::Const(bn_name+"_variance")
        .set_attr_value(variance_weight_tensor);

     //构建bn算子
    auto batchnorm = op::BatchNorm(bn_name)
        .set_input_x(data)
        .set_input_offset(bn_offset)
        .set_input_scale(bn_scale)  //设置输入Scalar
        .set_input_mean(bn_mean)    //设置输入均值
        .set_input_variance(bn_variance)    //设置输入方差
        .set_attr_data_format("NCHW")    //设置输入数据的格式NCHW
        .set_attr_is_training(false);   //此时非训练状态，设置成false

    TensorDesc batchnorm_input_desc_x(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
    TensorDesc batchnorm_output_desc_y(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
    //更新BN的输入信息
    batchnorm.update_input_desc_x(batchnorm_input_desc_x);
    batchnorm.update_input_desc_scale(batchnorm_input_desc_x);
    batchnorm.update_input_desc_offset(batchnorm_input_desc_x);
    batchnorm.update_input_desc_mean(batchnorm_input_desc_x);
    batchnorm.update_input_desc_variance(batchnorm_input_desc_x);

    batchnorm.update_output_desc_y(batchnorm_output_desc_y);
    batchnorm.update_output_desc_batch_mean(batchnorm_output_desc_y);
    batchnorm.update_output_desc_batch_variance(batchnorm_output_desc_y);

    return batchnorm;
}

/*
>> 生成Relu算子 
    relu_name：relu算子的名称
    data：上一个算子。进行构图的连接
*/
Operator GenReluOp(string relu_name,Operator data){

	// 因为relu算子接在bn算子后面，bn算子有多个输出，得指明是data为"y"的输出传入relu，防止因BN有多个输出造成图不明确
    auto relu = op::Relu(relu_name).set_input_x(data, "y");

    TensorDesc tensor_desc(ge::Shape(), FORMAT_ND, DT_FLOAT);
    relu.update_input_desc_x(tensor_desc);
    relu.update_output_desc_y(tensor_desc);
    return relu;
}

/*
>> 生成Maxpool算子
    pool_name：Maxpool算子的名称
    data：上一个算子。进行构图的连接
*/
Operator GenMaxpoolOp(string pool_name,Operator data){

    auto maxpool = op::MaxPoolV3(pool_name)
        .set_input_x(data)
        .set_attr_strides({1,1,2,2})  // 代表在四个维度（batch、 height,、width、channels）所移动的步长
        .set_attr_ksize({1,1,2,2}) //代表在四个维度（batch、 height,、width、channels）池化的尺寸，一般是[1, height, width, 1]
        .set_attr_pads({0,0,0,0})
        .set_attr_data_format("NCHW")
        .set_attr_padding_mode("CALCULATED")  //padding_mode默认CALCULATED，三种模式 "SAME" "VALID" or "CALCULATE"
        .set_attr_global_pooling(false)
        .set_attr_ceil_mode(false);  //是否在计算输出shape时，使用向上整取，默认false
    
    TensorDesc tensor_desc(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
    maxpool.update_input_desc_x(tensor_desc);
    maxpool.update_output_desc_y(tensor_desc);   
    return maxpool;
}

/*
>> 生成Flatten算子
    flatten_name：Flatten算子的名称
    data：上一个算子。进行构图的连接
*/
Operator GenFlattenOp(string flatten_name,Operator data){
    //构建Flatten算子
    auto flatten = op::FlattenV2(flatten_name).set_input_x(data);
    //更新算子输入输出信息
    TensorDesc tensor_desc(ge::Shape(), FORMAT_ND, DT_FLOAT);
    flatten.update_input_desc_x(tensor_desc);
    flatten.update_output_desc_y(tensor_desc); 

    return flatten;
}

/*
>> 生成Dense算子
    input_channel：输入通道数，MatMul的weight的shape为(out_channels, in_channels)
    output_channel：输出通道数
    dense_name：Dense算子的名称
    data：上一个算子，即被flatten后的输入数据，作为MatMul运算中的X。进行构图的连接
*/
Operator GenDenseOp(uint32_t input_channel,uint32_t output_channel,string dense_name,Operator data){


    // 构造dense层的权重矩阵，权重来自bin文件
    auto matmul_weight_shape = ge::Shape({output_channel, input_channel});
    TensorDesc desc_matmul_weight(matmul_weight_shape, FORMAT_ND, DT_FLOAT);
    Tensor matmul_weight_tensor(desc_matmul_weight);
    uint32_t matmul_weight_len = matmul_weight_shape.GetShapeSize() * sizeof(float);
    bool res = GetConstTensorFromBin(kPath + dense_name+".weight.bin", matmul_weight_tensor, matmul_weight_len);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
    }
    //构造matmul算子的权重常量算子
    auto matmul_weight = op::Const(dense_name+"_weight")
        .set_attr_value(matmul_weight_tensor);

    //构造偏重常量算子，读取偏置参数，作为OPTIONAL_INPUT的bias输入
    auto bias_add_shape = ge::Shape({ output_channel });
    TensorDesc desc_bias_add_const(bias_add_shape, FORMAT_ND, DT_FLOAT);
    Tensor bias_add_const_tensor(desc_bias_add_const);
    uint32_t bias_add_const_len = bias_add_shape.GetShapeSize() * sizeof(float);
    res = GetConstTensorFromBin(kPath + dense_name+".bias.bin", bias_add_const_tensor, bias_add_const_len);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
    }

    auto bias_add_const = op::Const(dense_name+"_bias")
        .set_attr_value(bias_add_const_tensor);
    
    // 构造MatMulV2算子，三个输入，权重矩阵W，flatten后的输入数据X，偏置bias
    auto matmul = op::MatMulV2(dense_name+"_matmul")
        .set_input_x1(data)
        .set_input_x2(matmul_weight)
	.set_attr_transpose_x2(true)
        .set_input_bias(bias_add_const);

    // 更新算子描述信息
    TensorDesc tensor_desc_matmul(ge::Shape(), FORMAT_ND, DT_FLOAT);
    matmul.update_input_desc_x1(tensor_desc_matmul);
    matmul.update_input_desc_x2(tensor_desc_matmul);
    matmul.update_input_desc_bias(tensor_desc_matmul);
    matmul.update_output_desc_y(tensor_desc_matmul);

    return matmul;
}

/*
>> 生成SoftmaxV2算子
    softmax_name：Flatten算子的名称
    data：上一个算子。进行构图的连接
*/
Operator GenSoftmaxOp(string flatten_name, Operator data){
    auto softmax = op::SoftmaxV2(flatten_name).set_input_x(data);   //softmax默认axes为-1
    //auto softmax = op::Softmax(flatten_name).set_input_x(data);   //softmax默认axes为-1
    return softmax;
}

//算子构图
bool GenGraph(Graph& graph)
{
    auto shape_data = vector<int64_t>({1,3,224,224});//输入数据[N,C,W,H],推理时batchsize为1
    TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NCHW, DT_FLOAT); //定义算子信息描述，shape传入desc_data

    // 实例化Data算子，名为data
    auto data = op::Data("data");
    data.update_input_desc_x(desc_data); //更新data1算子的输入数据信息描述，定义输入数据的shape，format和dtype
    data.update_output_desc_y(desc_data); //更新data1算子的输出数据信息描述，定义输入数据的shape，format和dtype

    
    //构建conv算子，weight_shape: 卷积核的shape格式：[filter_height, filter_width, in_channels / groups, out_channels]. 
    auto layer1_conv1 = GenConv2dOp(ge::Shape({64,3,3,3}),"layer1_conv1",data);
    auto layer1_bn1 = GenBNOp(ge::Shape({64}),"layer1_bn1",layer1_conv1);
    auto layer1_relu1 = GenReluOp("layer1_relu1",layer1_bn1);
    auto layer1_conv2 = GenConv2dOp(ge::Shape({64,64,3,3}),"layer1_conv2",layer1_relu1);
    auto layer1_bn2 = GenBNOp(ge::Shape({64}),"layer1_bn2",layer1_conv2);
    auto layer1_relu2 = GenReluOp("layer1_relu2",layer1_bn2);
    auto layer1_maxpool = GenMaxpoolOp("layer1_maxpool",layer1_relu2);

    /*
    TO-DO：
        1. 参考上述已经搭建好的输入算子和部分VGG17的算子构图部分，请按照VGG17的网络图，同学们自行完成构图的剩余部分的搭建
        注意：要包含最后的softmax层
    */

    // 输入算子
    std::vector<Operator> inputs{ data };
    // 输出算子
    std::vector<Operator> outputs{pred};
    
    graph.SetInputs(inputs).SetOutputs(outputs);

    return true;
}



int main(int argc, char* argv[])
{
    cout << "========== Test Start ==========" << endl;
    if (argc != kArgsNum) {
        cout << "[ERROR]input arg num must be 3! " << endl;
        cout << "The second arg stand for soc version! Please retry with your soc version " << endl;
        cout << "[Notice] Supported soc version as list:Ascend310 Ascend910 Ascend610 Ascend620 Hi3796CV300ES Hi3796CV300CS" << endl;
        cout << "The third arg stand for Generate Graph Options! Please retry with your soc version " << endl;
        cout << "[Notice] Supported Generate Graph Options as list:" << endl;
        cout << "    [gen]: GenGraph" << endl;
        cout << "    [tf]: generate from tensorflow origin model;" << endl;
        cout << "    [caffe]: generate from caffe origin model" << endl;
        return -1;
    }
    cout << argv[kSocVersion] << endl;
    cout << argv[kGenGraphOpt] << endl;

    // 1. 算子生成图 Genetate graph
    Graph graph1("IrGraph1");
    bool ret;

    if (string(argv[kGenGraphOpt]) == "gen") {
        ret = GenGraph(graph1);
        if (!ret) {
            cout << "========== Generate Graph1 Failed! ==========" << endl;
            return -1;
        }
        else {
            cout << "========== Generate Graph1 Success! ==========" << endl;
        }
    } 
   

    // 2.通过aclgrphBuildInitialize接口进行系统初始化，并申请资源,
    // 通过传入global_options参数配置离线模型编译初始化信息，soc_version 指定目标芯片版本; 
    std::map<AscendString, AscendString> global_options = {
        {AscendString(ge::ir_option::SOC_VERSION), AscendString(argv[kSocVersion])}  ,
    };
    auto status = aclgrphBuildInitialize(global_options);
    
    
    // 3. 通过aclgrphBuildModel接口将Graph编译为离线模型
    ModelBufferData model1;
    std::map<AscendString, AscendString> options;
    status = aclgrphBuildModel(graph1, options, model1);
    if (status == GRAPH_SUCCESS) {
        cout << "Build Model1 SUCCESS!" << endl;
    }
    else {
        cout << "Build Model1 Failed!" << endl;
    }


    // 4. 以通过aclgrphSaveModel将内存缓冲区中的模型保存为离线模型文件
    status = aclgrphSaveModel("ir_build_vgg17_builtin", model1);
    if (status == GRAPH_SUCCESS) {
        cout << "Save Offline Model1 SUCCESS!" << endl;
    }
    else {
        cout << "Save Offline Model1 Failed!" << endl;
    }

    // 5. 构图进程结束时，通过aclgrphBuildFinalize接口释放资源
    aclgrphBuildFinalize();
    return 0;
}
