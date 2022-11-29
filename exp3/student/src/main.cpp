// ---------------- 包含头文件 ----------------
#include "acl/acl.h"
#include <iostream>
#include <fstream>
#include <time.h>
#include <cstring>
#include <map>

using namespace std;

// ---------------- 资源初始化 ----------------
// TODO 参考实验手册完善 InitResource 函数
void InitResource(){}

// ---------------- 模型加载 ----------------
// TODO 参考实验手册完善 LoadModel 函数
void LoadModel(const char* modelPath){}


// ---------------- 图片读取 ----------------
// TODO 参考实验手册完善 ReadPictureTotHost、CopyDataFromHostToDevice、LoadPicture 函数
void ReadPictureTotHost(const char *picturePath){}

//申请Device侧的内存，再以内存复制的方式将内存中的图片数据传输到Device
void CopyDataFromHostToDevice(){}

void LoadPicture(const char* picturePath){}


// ---------------- 推理函数 ----------------
// TODO 参考实验手册完善 CreateModelInput、CreateModelOutput、Inference 函数
void CreateModelInput(){}

// 准备模型推理的输出数据结构
void CreateModelOutput(){}

// 执行模型
void Inference(){}

// ---------------- 推理结果处理 ----------------
// TODO 参考实验手册完善 PrintResult 函数
void PrintResult(){}

// ---------------- 模型卸载 ----------------
// TODO 参考实验手册完善 UnloadModel 函数
void UnloadModel(){}


// ---------------- 释放内存 ----------------
// TODO 参考实验手册完善 UnloadPicture 函数
void UnloadPicture(){}

// ---------------- 资源去初始化 ----------------
// TODO 参考实验手册完善 DestroyResource 函数
void DestroyResource(){}


void InferOnePic(const char* picturePath)
{
    printf("\nStart Inference %s\n", picturePath);
    // 3.定义一个读图片数据的函数，将测试图片数据读入内存，并传输到Device侧，用于后续推理使用
    LoadPicture(picturePath);

    // 4.定义一个推理的函数，用于执行推理
    clock_t start = clock();            // 推理开始时间
	Inference();

    // 5.定义一个推理结果数据处理的函数，用于在终端上屏显测试图片的置信度的类别编号
	PrintResult();
    clock_t finish = clock();           // 推理结束时间
    double duration = (double)(finish - start)*1000 / CLOCKS_PER_SEC;
    printf("Infer cost time: %f ms \n", duration);

}
// ---------------- 主函数 ----------------
int main()
{

    // 1.定义一个资源初始化的函数，用于AscendCL初始化、运行管理资源申请（指定计算设备）
	InitResource();

    // 2.定义一个模型加载的函数，加载图片分类的模型，用于后续推理使用
    const char *modelPath = "./model/vgg16.om";
    // const char *modelPath = "./model/vgg16_fp32.om";
	LoadModel(modelPath);

    InferOnePic("./data/daisy_demo.bin");
    InferOnePic("./data/roses_demo.bin");
    InferOnePic("./data/sunflowers_demo.bin");
    InferOnePic("./data/tulips_demo.bin");

    // 6.定义一个模型卸载的函数，卸载图片分类的模型
	UnloadModel();

    // 7.定义一个函数，用于释放内存、销毁推理相关的数据类型，防止内存泄露
	UnloadPicture();

    // 8.定义一个资源去初始化的函数，用于AscendCL去初始化、运行管理资源释放（释放计算设备）
	DestroyResource();

	return 1;
}