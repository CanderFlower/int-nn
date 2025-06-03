#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include "intnn_mat.h"
#include "intnn_loader.h"

#define TEST_ASSERT(cond, msg)        \
    if (!(cond)) {                    \
        printf("[FAILED] %s\n", msg); \
        exit(1);                      \
    } else {                          \
        printf("[PASSED] %s\n", msg); \
    }

void test_mnist_images() {
    printf("=== Testing MNIST image loader ===\n");

    // 加载前 5 张训练图像，MNIST 每张 28x28=784 像素，放到 (5 x 784) 的矩阵里
    intnn_mat* images = intnn_create_mat(0,0);
    intnn_load_mnist_images(images, 5, true);

    // 检查第一张图的前几个像素值（可自行 eyeball 文件数据）
    int pixel00 = intnn_get_elem(images, 0, 0);
    int pixel10 = intnn_get_elem(images, 1, 0);
    printf("First pixel values: img0[0]=%d, img1[0]=%d\n", pixel00, pixel10);

    intnn_free_mat(images);
}

void test_mnist_labels() {
    printf("=== Testing MNIST label loader ===\n");

    // 加载前 5 个训练标签，放到 (5 x 1) 的矩阵
    intnn_mat* labels = intnn_create_mat(0,0);
    intnn_load_mnist_labels(labels, 5, true);


    // 验证标签值在 [0,9] 范围内
    for (int i = 0; i < 5; i++) {
        int l = intnn_get_elem(labels, i, 0);
        TEST_ASSERT(l >= 0 && l <= 9, "label in [0,9]");
    }

    intnn_free_mat(labels);
}

void test_fashion_mnist_images() {
    printf("=== Testing Fashion-MNIST image loader ===\n");
    intnn_mat* images = intnn_create_mat(0,0);
    intnn_load_fashion_mnist_images(images, 3, true);

    intnn_free_mat(images);
}

void test_fashion_mnist_labels() {
    printf("=== Testing Fashion-MNIST label loader ===\n");
    intnn_mat* labels = intnn_create_mat(0,0);
    intnn_load_fashion_mnist_labels(labels, 3, true);


    intnn_free_mat(labels);
}

int main() {
    printf("==== Running MNIST & Fashion-MNIST tests ===\n");
    test_mnist_images();
    test_mnist_labels();
    test_fashion_mnist_images();
    test_fashion_mnist_labels();
    printf("All MNIST tests passed.\n");
    return 0;
}
