#include "intnn_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

// Windows 下使用 URLDownloadToFileA 需要链接 urlmon.lib
#ifdef _WIN32
#include <windows.h>
#include <urlmon.h>
#endif

// ------------------------------
// 工具函数
// ------------------------------

bool intnn_file_exists(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

void intnn_download_dataset(intnn_dataset_type dataset) {
    (void)dataset;
    // 目前只针对 diabetes 数据集做下载示例
    const char* fileName = "dataset/diabetes.tab.txt";
    if (intnn_file_exists(fileName)) {
        printf("File already exists: %s\n", fileName);
        return;
    }
#ifdef _WIN32
    // Windows 下调用 URLDownloadToFileA
    HRESULT hr = URLDownloadToFileA(
        NULL,
        "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt",
        "dataset/diabetes.tab.txt",
        0,
        NULL
    );
    if (hr != S_OK) {
        printf("Failed to download %s\n", fileName);
    }
#else
    // Linux/Unix 下调用 curl（需要系统已安装 curl）
    char cmd[512];
    // 确保 dataset 目录存在
    snprintf(cmd, sizeof(cmd),
             "mkdir -p dataset && curl -L -o dataset/diabetes.tab.txt "
             "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt");
    int ret = system(cmd);
    if (ret != 0) {
        printf("Failed to download %s via curl\n", fileName);
    }
#endif
}

int intnn_reverse_int(int i) {
    uint32_t v = (uint32_t)i;
    uint32_t ch1 = (v & 0x000000FF);
    uint32_t ch2 = (v & 0x0000FF00) >> 8;
    uint32_t ch3 = (v & 0x00FF0000) >> 16;
    uint32_t ch4 = (v & 0xFF000000) >> 24;
    return (int)((ch1 << 24) | (ch2 << 16) | (ch3 << 8) | ch4);
}

// ------------------------------
// CSV 加载
// ------------------------------

void intnn_load_csv(intnn_mat* outMat, const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("%s does not exist!\n", filename);
        return;
    }

    char buffer[1024];
    // 读取 header 行并丢弃
    if (!fgets(buffer, sizeof(buffer), fp)) {
        fclose(fp);
        return;
    }
    // 计算列数（逗号分隔）
    int numCols = 0;
    {
        char* tmp = strdup(buffer);
        char* token = strtok(tmp, ",");
        while (token) {
            numCols++;
            token = strtok(NULL, ",");
        }
        free(tmp);
    }
    // 计算行数
    int numRows = 0;
    while (fgets(buffer, sizeof(buffer), fp)) {
        numRows++;
    }
    printf("Rows, Cols: %d, %d\n", numRows, numCols);

    // 回到文件开头，跳过 header
    fseek(fp, 0, SEEK_SET);
    fgets(buffer, sizeof(buffer), fp);

    intnn_reset_zero(outMat, numRows, numCols);

    int r = 0;
    while (fgets(buffer, sizeof(buffer), fp)) {
        int c = 0;
        char* line = strdup(buffer);
        char* token = strtok(line, ",");
        while (token && c < numCols) {
            int val = atoi(token);
            intnn_set_elem(outMat, r, c, val);
            c++;
            token = strtok(NULL, ",");
        }
        free(line);
        r++;
    }
    fclose(fp);
}

// ------------------------------
// Diabetes 数据集解析（制表符分隔）
// ------------------------------

void intnn_parse_dataset_diabetes(intnn_mat* outMat, const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Failed to open %s\n", filename);
        return;
    }

    char buffer[1024];
    // 读取 header 行并丢弃
    if (!fgets(buffer, sizeof(buffer), fp)) {
        fclose(fp);
        return;
    }
    // 计算列数（制表符分隔）
    int numCols = 0;
    {
        char* tmp = strdup(buffer);
        char* token = strtok(tmp, "\t");
        while (token) {
            numCols++;
            token = strtok(NULL, "\t");
        }
        free(tmp);
    }
    // 计算行数
    int numRows = 0;
    while (fgets(buffer, sizeof(buffer), fp)) {
        numRows++;
    }

    // 回到文件开头，跳过 header
    fseek(fp, 0, SEEK_SET);
    fgets(buffer, sizeof(buffer), fp);

    intnn_reset_zero(outMat, numRows, numCols);

    int r = 0;
    while (fgets(buffer, sizeof(buffer), fp)) {
        int c = 0;
        char* line = strdup(buffer);
        char* token = strtok(line, "\t");
        while (token && c < numCols) {
            int val = atoi(token);
            intnn_set_elem(outMat, r, c, val);
            c++;
            token = strtok(NULL, "\t");
        }
        free(line);
        r++;
    }
    fclose(fp);
}

// ------------------------------
// 加载 MNIST/Fashion-MNIST 图像
// ------------------------------

// 辅助：从 idx3-ubyte 文件加载 numToLoad 张图像到 outMat
static void intnn_load_idx3_images(intnn_mat* outMat, const char* filepath, int numToLoad) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        printf("Failed to open %s\n", filepath);
        return;
    }
    // 读 header：魔数、项目数、行数、列数（4 个 32bit big-endian）
    int magic = 0, numItems = 0, numRows = 0, numCols = 0;
    fread(&magic, sizeof(int), 1, fp);
    fread(&numItems, sizeof(int), 1, fp);
    fread(&numRows, sizeof(int), 1, fp);
    fread(&numCols, sizeof(int), 1, fp);
    magic = intnn_reverse_int(magic);
    numItems = intnn_reverse_int(numItems);
    numRows = intnn_reverse_int(numRows);
    numCols = intnn_reverse_int(numCols);

    printf("Loading %s: magic=%d, items=%d, rows=%d, cols=%d\n",
           filepath, magic, numItems, numRows, numCols);

    // 初始化 outMat: numToLoad x (numRows*numCols)
    intnn_reset_zero(outMat, numToLoad, numRows * numCols);

    for (int i = 0; i < numToLoad; i++) {
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                unsigned char pixel = 0;
                fread(&pixel, sizeof(unsigned char), 1, fp);
                intnn_set_elem(outMat, i, r * numCols + c, (int)pixel);
            }
        }
    }
    fclose(fp);
}

void intnn_load_mnist_images(intnn_mat* outMat, int numImagesToLoad, bool isTrain) {
    const char* filepath = isTrain
        ? "dataset/mnist/train-images.idx3-ubyte"
        : "dataset/mnist/t10k-images.idx3-ubyte";
    intnn_load_idx3_images(outMat, filepath, numImagesToLoad);
}

void intnn_load_fashion_mnist_images(intnn_mat* outMat, int numImagesToLoad, bool isTrain) {
    const char* filepath = isTrain
        ? "dataset/fashion_mnist/train-images-idx3-ubyte"
        : "dataset/fashion_mnist/t10k-images-idx3-ubyte";
    intnn_load_idx3_images(outMat, filepath, numImagesToLoad);
}

// ------------------------------
// 加载 MNIST/Fashion-MNIST 标签
// ------------------------------

static void intnn_load_idx1_labels(intnn_mat* outMat, const char* filepath, int numToLoad) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        printf("Failed to open %s\n", filepath);
        return;
    }
    // 读 header：魔数、项目数（32bit 大端）
    int magic = 0, numItems = 0;
    fread(&magic, sizeof(int), 1, fp);
    fread(&numItems, sizeof(int), 1, fp);
    magic = intnn_reverse_int(magic);
    numItems = intnn_reverse_int(numItems);

    printf("Loading %s: magic=%d, items=%d\n", filepath, magic, numItems);

    intnn_reset_zero(outMat, numToLoad, 1);

    for (int i = 0; i < numToLoad; i++) {
        unsigned char label = 0;
        fread(&label, sizeof(unsigned char), 1, fp);
        intnn_set_elem(outMat, i, 0, (int)label);
    }
    fclose(fp);
}

void intnn_load_mnist_labels(intnn_mat* outMat, int numLabelsToLoad, bool isTrain) {
    const char* filepath = isTrain
        ? "dataset/mnist/train-labels.idx1-ubyte"
        : "dataset/mnist/t10k-labels.idx1-ubyte";
    intnn_load_idx1_labels(outMat, filepath, numLabelsToLoad);
}

void intnn_load_fashion_mnist_labels(intnn_mat* outMat, int numLabelsToLoad, bool isTrain) {
    const char* filepath = isTrain
        ? "dataset/fashion_mnist/train-labels-idx1-ubyte"
        : "dataset/fashion_mnist/t10k-labels-idx1-ubyte";
    intnn_load_idx1_labels(outMat, filepath, numLabelsToLoad);
}