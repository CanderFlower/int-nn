#ifndef INTNN_LOADER_H
#define INTNN_LOADER_H

#include <stdbool.h>

#include "intnn_mat.h"
#include "intnn_mat3d.h"

#ifdef __cplusplus
extern "C" {
#endif

// 数据集枚举
typedef enum {
    INTNN_DATASET_DIABETES,
    INTNN_DATASET_MNIST
} intnn_dataset_type;

// 工具函数
bool intnn_file_exists(const char* filename);
// void intnn_download_dataset(intnn_dataset_type dataset);

// CSV 加载
void intnn_load_csv(intnn_mat* outMat, const char* filename);

// Diabetes 数据集特化解析
void intnn_parse_dataset_diabetes(intnn_mat* outMat, const char* filename);

// MNIST & Fashion-MNIST 加载
void intnn_load_mnist_images(intnn_mat* outMat, int numImagesToLoad, bool isTrain);
void intnn_load_mnist_labels(intnn_mat* outMat, int numLabelsToLoad, bool isTrain);
void intnn_load_fashion_mnist_images(intnn_mat* outMat, int numImagesToLoad, bool isTrain);
void intnn_load_fashion_mnist_labels(intnn_mat* outMat, int numLabelsToLoad, bool isTrain);

// 工具函数
int intnn_reverse_int(int i);

#ifdef __cplusplus
}
#endif

#endif // INTNN_LOADER_H
