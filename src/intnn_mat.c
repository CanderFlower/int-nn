#include "intnn_mat.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 创建矩阵，所有元素初始化为0
intnn_mat* intnn_create_mat(int rows, int cols) {
    if (rows <= 0 || cols <= 0)
        return NULL;

    intnn_mat* mat = (intnn_mat*)malloc(sizeof(intnn_mat));
    if (!mat)
        return NULL;

    mat->mRows = rows;
    mat->mCols = cols;
    mat->mDeleteOnDestruct = true;
    mat->mName = NULL;

    mat->mMat = (int**)malloc(rows * sizeof(int*));
    if (!mat->mMat) {
        free(mat);
        return NULL;
    }

    for (int r = 0; r < rows; ++r) {
        mat->mMat[r] = (int*)calloc(cols, sizeof(int));
        if (!mat->mMat[r]) {
            // 释放已分配行
            for (int i = 0; i < r; ++i)
                free(mat->mMat[i]);
            free(mat->mMat);
            free(mat);
            return NULL;
        }
    }

    return mat;
}

// 释放矩阵
void intnn_free_mat(intnn_mat* mat) {
    if (!mat)
        assert(0);
    if (mat->mDeleteOnDestruct) {
        if (mat->mMat) {
            for (int r = 0; r < mat->mRows; ++r) {
                free(mat->mMat[r]);
            }
            free(mat->mMat);
        }
    }
    //free(mat);
}

// 复制矩阵
intnn_mat* intnn_copy_mat(const intnn_mat* src) {
    if (!src)
        return NULL;
    intnn_mat* dst = intnn_create_mat(src->mRows, src->mCols);
    if (!dst)
        return NULL;

    for (int r = 0; r < src->mRows; ++r) {
        memcpy(dst->mMat[r], src->mMat[r], sizeof(int) * src->mCols);
    }
    dst->mName = src->mName;  // 指针复制，不拷贝字符串内容
    return dst;
}

// 设置是否自动释放数据
void intnn_set_delete_on_destruct(intnn_mat* mat, bool flag) {
    if (!mat)
        assert(0);
    mat->mDeleteOnDestruct = flag;
}

// 设置所有元素为常数
void intnn_set_all_constant(intnn_mat* mat, int value) {
    if (!mat)
        assert(0);
    for (int r = 0; r < mat->mRows; ++r) {
        for (int c = 0; c < mat->mCols; ++c) {
            mat->mMat[r][c] = value;
        }
    }
}

// 设置随机元素，范围[minVal, maxVal]，allowZero决定是否允许0
void intnn_set_random(intnn_mat* mat, bool allowZero, int minVal, int maxVal) {
    if (!mat || minVal > maxVal)
        assert(0);
    for (int r = 0; r < mat->mRows; ++r) {
        for (int c = 0; c < mat->mCols; ++c) {
            int val;
            do {
                val = minVal + rand() % (maxVal - minVal + 1);
            } while (!allowZero && val == 0);
            mat->mMat[r][c] = val;
        }
    }
}

// 设置单个元素
void intnn_set_elem(intnn_mat* mat, int r, int c, int val) {
    if (!mat)
        assert(0);
    if (r < 0 || r >= mat->mRows || c < 0 || c >= mat->mCols)
        assert(0);
    mat->mMat[r][c] = val;
}

// 从一维数组创建矩阵（按行优先）
intnn_mat* intnn_set_mat_from_array(int rows, int cols, int* data) {
    if (rows <= 0 || cols <= 0 || !data)
        return NULL;
    intnn_mat* mat = intnn_create_mat(rows, cols);
    if (!mat)
        return NULL;

    for (int r = 0; r < rows; ++r) {
        memcpy(mat->mMat[r], &data[r * cols], sizeof(int) * cols);
    }
    return mat;
}

// 重新设置矩阵大小并清零（旧数据释放）
void intnn_reset_zero(intnn_mat* mat, int rows, int cols) {
    if (!mat || rows <= 0 || cols <= 0)
        assert(0);
    if (mat->mDeleteOnDestruct && mat->mMat) {
        for (int r = 0; r < mat->mRows; ++r)
            free(mat->mMat[r]);
        free(mat->mMat);
    }
    mat->mRows = rows;
    mat->mCols = cols;
    mat->mMat = (int**)malloc(rows * sizeof(int*));
    for (int r = 0; r < rows; ++r) {
        mat->mMat[r] = (int*)calloc(cols, sizeof(int));
    }
    mat->mDeleteOnDestruct = true;
}

// 重新设置矩阵大小，所有元素设为1
void intnn_reset_all_ones(intnn_mat* mat, int rows, int cols) {
    intnn_reset_zero(mat, rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            mat->mMat[r][c] = 1;
}

// 获取行数
int intnn_rows(const intnn_mat* mat) {
    if (!mat)
        return 0;
    return mat->mRows;
}

// 获取列数
int intnn_cols(const intnn_mat* mat) {
    if (!mat)
        return 0;
    return mat->mCols;
}

// 求和所有元素
int intnn_sum(const intnn_mat* mat) {
    if (!mat)
        return 0;
    int total = 0;
    for (int r = 0; r < mat->mRows; ++r)
        for (int c = 0; c < mat->mCols; ++c)
            total += mat->mMat[r][c];
    return total;
}

// 元素总数
int intnn_num_elems(const intnn_mat* mat) {
    if (!mat)
        return 0;
    return mat->mRows * mat->mCols;
}

// 获取单个元素
int intnn_get_elem(const intnn_mat* mat, int r, int c) {
    if (!mat)
        return 0;
    if (r < 0 || r >= mat->mRows || c < 0 || c >= mat->mCols)
        return 0;
    return mat->mMat[r][c];
}

// 获取内部二维数据指针（非const，谨慎使用）
int** intnn_get_data(intnn_mat* mat) {
    if (!mat)
        assert(0);
    return mat->mMat;
}

// 判断两个矩阵尺寸是否相等
bool intnn_dims_equal(const intnn_mat* m1, const intnn_mat* m2) {
    if (!m1 || !m2)
        assert(0);
    return (m1->mRows == m2->mRows) && (m1->mCols == m2->mCols);
}

// 判断矩阵尺寸是否等于指定大小
bool intnn_dims_equal_size(const intnn_mat* mat, int r, int c) {
    if (!mat)
        assert(0);
    return (mat->mRows == r) && (mat->mCols == c);
}

// 获取指定行最大值的索引
int intnn_get_max_index_in_row(const intnn_mat* mat, int r) {
    if (!mat)
        return -1;
    if (r < 0 || r >= mat->mRows)
        return -1;

    int maxIndex = 0;
    int maxVal = mat->mMat[r][0];
    for (int c = 1; c < mat->mCols; ++c) {
        if (mat->mMat[r][c] > maxVal) {
            maxVal = mat->mMat[r][c];
            maxIndex = c;
        }
    }
    return maxIndex;
}

// 获取指定行最小值
int intnn_get_row_min(const intnn_mat* mat, int r) {
    if (!mat)
        return 0;
    if (r < 0 || r >= mat->mRows)
        return 0;

    int minVal = mat->mMat[r][0];
    for (int c = 1; c < mat->mCols; ++c) {
        if (mat->mMat[r][c] < minVal) {
            minVal = mat->mMat[r][c];
        }
    }
    return minVal;
}

// 获取指定行最大值
int intnn_get_row_max(const intnn_mat* mat, int r) {
    if (!mat)
        return 0;
    if (r < 0 || r >= mat->mRows)
        return 0;

    int maxVal = mat->mMat[r][0];
    for (int c = 1; c < mat->mCols; ++c) {
        if (mat->mMat[r][c] > maxVal) {
            maxVal = mat->mMat[r][c];
        }
    }
    return maxVal;
}

// 获取指定列最小值
int intnn_get_col_min(const intnn_mat* mat, int c) {
    if (!mat)
        return 0;
    if (c < 0 || c >= mat->mCols)
        return 0;

    int minVal = mat->mMat[0][c];
    for (int r = 1; r < mat->mRows; ++r) {
        if (mat->mMat[r][c] < minVal) {
            minVal = mat->mMat[r][c];
        }
    }
    return minVal;
}

// 获取指定列最大值
int intnn_get_col_max(const intnn_mat* mat, int c) {
    if (!mat)
        return 0;
    if (c < 0 || c >= mat->mCols)
        return 0;

    int maxVal = mat->mMat[0][c];
    for (int r = 1; r < mat->mRows; ++r) {
        if (mat->mMat[r][c] > maxVal) {
            maxVal = mat->mMat[r][c];
        }
    }
    return maxVal;
}

// 计算矩阵所有元素的平均值（整除）
int intnn_average(const intnn_mat* mat) {
    if (!mat || mat->mRows == 0 || mat->mCols == 0)
        return 0;
    long long sum = 0;  // 用 long long 防止溢出
    for (int r = 0; r < mat->mRows; r++) {
        for (int c = 0; c < mat->mCols; c++) {
            sum += mat->mMat[r][c];
        }
    }
    return (int)(sum / (mat->mRows * mat->mCols));
}

// 根据已有平均值计算方差
int intnn_variance_with_avg(const intnn_mat* mat, int avg) {
    if (!mat || mat->mRows == 0 || mat->mCols == 0)
        return 0;
    long long sum_var = 0;
    for (int r = 0; r < mat->mRows; r++) {
        for (int c = 0; c < mat->mCols; c++) {
            int diff = mat->mMat[r][c] - avg;
            sum_var += (long long)diff * diff;
        }
    }
    return (int)(sum_var / (mat->mRows * mat->mCols));
}

// 计算方差（内部调用上面，先算平均）
int intnn_variance(const intnn_mat* mat) {
    int avg = intnn_average(mat);
    return intnn_variance_with_avg(mat, avg);
}

// 根据平均值计算标准差
int intnn_stdev_with_avg(const intnn_mat* mat, int avg) {
    int var = intnn_variance_with_avg(mat, avg);
    return intnn_floor_sqrt(var);
}

// 计算标准差（先求平均）
int intnn_stdev(const intnn_mat* mat) {
    int avg = intnn_average(mat);
    return intnn_stdev_with_avg(mat, avg);
}

// 按列计算平均，替换矩阵为每列平均值的一行矩阵（行=1）
void intnn_average_colwise(intnn_mat* mat) {
    if (!mat || mat->mRows == 0 || mat->mCols == 0)
        assert(0);

    for (int c = 0; c < mat->mCols; c++) {
        long long sum = 0;
        for (int r = 0; r < mat->mRows; r++) {
            sum += mat->mMat[r][c];
        }
        int avg = (int)(sum / mat->mRows);
        for (int r = 0; r < mat->mRows; r++) {
            mat->mMat[r][c] = avg;
        }
    }
}

// 标准化，将矩阵元素映射到 [low, high]，基于 numSigma 个标准差范围
void intnn_standardize(intnn_mat* mat, int numSigma, int low, int high) {
    if (!mat || mat->mRows == 0 || mat->mCols == 0)
        assert(0);
    int avg = intnn_average(mat);
    int std = intnn_stdev_with_avg(mat, avg);
    int range = 2 * numSigma * std;
    if (range == 0)
        range = 1;  // 防止除零
    int scale = (high - low);

    for (int r = 0; r < mat->mRows; r++) {
        for (int c = 0; c < mat->mCols; c++) {
            int val = mat->mMat[r][c];
            int standardized = ((val - avg) * scale) / range + low;
            if (standardized < low)
                standardized = low;
            else if (standardized > high)
                standardized = high;
            mat->mMat[r][c] = standardized;
        }
    }
}

// 按行归一化到 [newMin, newMax]
void intnn_normalize_rowwise(intnn_mat* mat, int newMin, int newMax) {
    if (!mat || mat->mRows == 0 || mat->mCols == 0)
        assert(0);
    for (int r = 0; r < mat->mRows; r++) {
        int minVal = mat->mMat[r][0], maxVal = mat->mMat[r][0];
        for (int c = 1; c < mat->mCols; c++) {
            int val = mat->mMat[r][c];
            if (val < minVal)
                minVal = val;
            if (val > maxVal)
                maxVal = val;
        }
        int diff = maxVal - minVal;
        if (diff == 0)
            diff = 1;  // 防止除零
        for (int c = 0; c < mat->mCols; c++) {
            int val = mat->mMat[r][c];
            mat->mMat[r][c] =
                ((val - minVal) * (newMax - newMin)) / diff + newMin;
        }
    }
}

// 按列归一化到 [newMin, newMax]
void intnn_normalize_colwise(intnn_mat* mat, int newMin, int newMax) {
    if (!mat || mat->mRows == 0 || mat->mCols == 0)
        assert(0);
    for (int c = 0; c < mat->mCols; c++) {
        int minVal = mat->mMat[0][c], maxVal = mat->mMat[0][c];
        for (int r = 1; r < mat->mRows; r++) {
            int val = mat->mMat[r][c];
            if (val < minVal)
                minVal = val;
            if (val > maxVal)
                maxVal = val;
        }
        int diff = maxVal - minVal;
        if (diff == 0)
            diff = 1;  // 防止除零
        for (int r = 0; r < mat->mRows; r++) {
            int val = mat->mMat[r][c];
            mat->mMat[r][c] =
                ((val - minVal) * (newMax - newMin)) / diff + newMin;
        }
    }
}

// 整个矩阵归一化（全局minmax）
void intnn_normalize_minmax(intnn_mat* mat, int newMin, int newMax) {
    if (!mat || mat->mRows == 0 || mat->mCols == 0)
        assert(0);
    int minVal = mat->mMat[0][0], maxVal = mat->mMat[0][0];
    for (int r = 0; r < mat->mRows; r++) {
        for (int c = 0; c < mat->mCols; c++) {
            int val = mat->mMat[r][c];
            if (val < minVal)
                minVal = val;
            if (val > maxVal)
                maxVal = val;
        }
    }
    int diff = maxVal - minVal;
    if (diff == 0)
        diff = 1;
    for (int r = 0; r < mat->mRows; r++) {
        for (int c = 0; c < mat->mCols; c++) {
            int val = mat->mMat[r][c];
            mat->mMat[r][c] =
                ((val - minVal) * (newMax - newMin)) / diff + newMin;
        }
    }
}

// 限制矩阵元素在区间 [low, high]
void intnn_clamp_mat(intnn_mat* mat, int low, int high) {
    if (!mat)
        assert(0);
    for (int r = 0; r < mat->mRows; r++) {
        for (int c = 0; c < mat->mCols; c++) {
            if (mat->mMat[r][c] < low)
                mat->mMat[r][c] = low;
            else if (mat->mMat[r][c] > high)
                mat->mMat[r][c] = high;
        }
    }
}

// 矩阵乘法：out = a * b
void intnn_mat_mul_mat(intnn_mat* out, const intnn_mat* a, const intnn_mat* b) {
    if (!out || !a || !b)
        assert(0);
    if (a->mCols != b->mRows)
    {
		printf("a size: (%d, %d)\n", a->mRows, a->mCols);
		printf("b size: (%d, %d)\n", b->mRows, b->mCols);
		printf("Matrix multiplication dimension mismatch: a.cols != b.rows\n");
        assert(0);
    }
    if (out->mRows != a->mRows || out->mCols != b->mCols){
        assert(0);
    }
    /*if (!(a->mRows < 11 || a->mCols < 11 || b->mRows < 11 || b->mCols < 11)) {
        printf("A:\n");
        for (int i = 0; i < 10; i++, printf("\n"))
            for (int j = 0; j < 10; j++)
                printf("%d ", a->mMat[i][j]);
        printf("B:\n");
        for (int i = 0; i < 10; i++, printf("\n"))
            for (int j = 0; j < 10; j++)
                printf("%d ", b->mMat[i][j]);
        printf("OUT OF MULTI:\n");
        for (int i = 0; i < 10; i++, printf("\n"))
            for (int j = 0; j < 10; j++)
                printf("%d ", out->mMat[i][j]);
    }
    printf("------------------\n");*/
    int flag = 1;
    for (int i = 0; i < a->mRows; i++)
        for (int j = 0; j < a->mCols; j++)
            if (a->mMat[i][j]) flag = 0;

    for (int r = 0; r < a->mRows; r++) {
        for (int c = 0; c < b->mCols; c++) {
            long long sum = 0;
            for (int k = 0; k < a->mCols; k++) {
                sum += (long long)a->mMat[r][k] * b->mMat[k][c];
                //if (sum) printf("%d %d %d\n", a->mMat[r][k], b->mMat[k][c], k);
            }
            out->mMat[r][c] = (int)sum;
        }
    }
    /*if (!(a->mRows < 11 || a->mCols < 11 || b->mRows < 11 || b->mCols < 11)) {
        printf("A:\n");
        for (int i = 0; i < 10; i++, printf("\n"))
            for (int j = 0; j < 10; j++)
                printf("%d ", a->mMat[i][j]);
        printf("B:\n");
        for (int i = 0; i < 10; i++, printf("\n"))
            for (int j = 0; j < 10; j++)
                printf("%d ", b->mMat[i][j]);
        printf("OUT OF MULTI:\n");
        for (int i = 0; i < 10; i++, printf("\n"))
            for (int j = 0; j < 10; j++)
                printf("%d ", out->mMat[i][j]);
    }*/
}

// 矩阵加法：out = a + b
void intnn_mat_add_mat(intnn_mat* out, const intnn_mat* a, const intnn_mat* b) {
    if (!out || !a || !b)
        assert(0);
    if (a->mRows != b->mRows || a->mCols != b->mCols)
        assert(0);
    if (out->mRows != a->mRows || out->mCols != a->mCols)
        assert(0);

    for (int r = 0; r < a->mRows; r++) {
        for (int c = 0; c < a->mCols; c++) {
            out->mMat[r][c] = a->mMat[r][c] + b->mMat[r][c];
        }
    }
}

// 元素乘法（Hadamard积）：out = a ⊙ b
void intnn_mat_elem_mul_mat(intnn_mat* out,
                            const intnn_mat* a,
                            const intnn_mat* b) {
    if (!a || !b || !out) assert(0);
    if (a->mRows != b->mRows || a->mCols != b->mCols)
        assert(0);

    
    for (int r = 0; r < a->mRows; r++) {
        for (int c = 0; c < a->mCols; c++) {
            out->mMat[r][c] = a->mMat[r][c] * b->mMat[r][c];
        }
    }
}

// 元素除法：out = a / b （整数除法，注意除零检查）
void intnn_mat_elem_div_mat(intnn_mat* out,
                            const intnn_mat* a,
                            const intnn_mat* b) {
    if (!out || !a || !b)
        assert(0);
    if (a->mRows != b->mRows || a->mCols != b->mCols)
        assert(0);
    if (out->mRows != a->mRows || out->mCols != a->mCols)
        assert(0);

    for (int r = 0; r < a->mRows; r++) {
        for (int c = 0; c < a->mCols; c++) {
            int denom = b->mMat[r][c];
            if (denom == 0)
                denom = 1;  // 防止除零
            out->mMat[r][c] = a->mMat[r][c] / denom;
        }
    }
}

// 矩阵加常数：out = a + val
void intnn_mat_add_const(intnn_mat* out, const intnn_mat* a, int val) {
    if (!out || !a)
        assert(0);
    if (out->mRows != a->mRows || out->mCols != a->mCols)
        assert(0);

    for (int r = 0; r < a->mRows; r++) {
        for (int c = 0; c < a->mCols; c++) {
            out->mMat[r][c] = a->mMat[r][c] + val;
        }
    }
}

// 矩阵乘常数：out = a * val
void intnn_mat_mul_const(intnn_mat* out, const intnn_mat* a, int val) {
    if (!out || !a)
        assert(0);
    if (out->mRows != a->mRows || out->mCols != a->mCols)
        assert(0);

    for (int r = 0; r < a->mRows; r++) {
        for (int c = 0; c < a->mCols; c++) {
            out->mMat[r][c] = a->mMat[r][c] * val;
        }
    }
}

// 矩阵除常数：out = a / val
void intnn_mat_div_const(intnn_mat* out, const intnn_mat* a, int val) {
    if (!out || !a)
        assert(0);
    if (val == 0)
        val = 1;  // 防止除零
    if (out->mRows != a->mRows || out->mCols != a->mCols)
        assert(0);

    for (int r = 0; r < a->mRows; r++) {
        for (int c = 0; c < a->mCols; c++) {
            out->mMat[r][c] = a->mMat[r][c] / val;
        }
    }
}

// 判断两个矩阵维度是否相同
static inline int dimsEqual(const intnn_mat* a, const intnn_mat* b) {
    return (a->mRows == b->mRows) && (a->mCols == b->mCols);
}

// 重置矩阵尺寸并清零
void resetZero(intnn_mat* mat, int rows, int cols) {
    if(mat) intnn_free_mat(mat);

    // 更新维度
    mat->mRows = rows;
    mat->mCols = cols;

    // 分配新内存
    mat->mMat = (int**)malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        mat->mMat[i] = (int*)calloc(cols, sizeof(int));
    }
}

void intnn_self_add_const(intnn_mat* mat, int val) {
    for (int r = 0; r < mat->mRows; ++r) {
        for (int c = 0; c < mat->mCols; ++c) {
            mat->mMat[r][c] += val;
        }
    }
}

void intnn_self_mul_const(intnn_mat* mat, int val) {
    for (int r = 0; r < mat->mRows; ++r) {
        for (int c = 0; c < mat->mCols; ++c) {
            mat->mMat[r][c] *= val;
        }
    }
}

void intnn_self_div_const(intnn_mat* mat, int val) {
    if (val == 0) {
        // 避免除零，报错或返回
        assert(0);
    }
    for (int r = 0; r < mat->mRows; ++r) {
        for (int c = 0; c < mat->mCols; ++c) {
            mat->mMat[r][c] /= val;
        }
    }
}

void intnn_self_elem_add_const(intnn_mat* mat, int r, int c, int val) {
    if (r >= 0 && r < mat->mRows && c >= 0 && c < mat->mCols) {
        mat->mMat[r][c] += val;
    }
}

void intnn_self_add_mat(intnn_mat* mat, const intnn_mat* b) {
    if (mat->mCols != b->mCols)
        assert(0);
    if (mat->mRows == b->mRows) {
        for (int r = 0; r < mat->mRows; ++r) {
            for (int c = 0; c < mat->mCols; ++c) {
                mat->mMat[r][c] += b->mMat[r][c];
            }
        }
    }
    else { // broadcast
        for (int r = 0; r < mat->mRows; ++r) {
            for (int c = 0; c < mat->mCols; ++c) {
                mat->mMat[r][c] += b->mMat[0][c];
            }
        }
    }
}

void intnn_self_sub_mat(intnn_mat* mat, const intnn_mat* b) {
    if (!dimsEqual(mat, b))
        assert(0);
    for (int r = 0; r < mat->mRows; ++r) {
        for (int c = 0; c < mat->mCols; ++c) {
            mat->mMat[r][c] -= b->mMat[r][c];
        }
    }
}
void intnn_self_mul_mat(intnn_mat* mat, const intnn_mat* b) {
    // 矩阵乘法： mat = mat * b
    // 先保存 mat 的副本
    if (mat->mCols != b->mRows)
        assert(0);  // 维度不匹配

    intnn_mat temp;
    // 假设你有分配和释放矩阵的函数
    // 这里简化，先分配 temp.mMat
    resetZero(&temp, mat->mRows, b->mCols);

    for (int r = 0; r < temp.mRows; ++r) {
        for (int c = 0; c < temp.mCols; ++c) {
            int sum = 0;
            for (int k = 0; k < mat->mCols; ++k) {
                sum += mat->mMat[r][k] * b->mMat[k][c];
            }
            temp.mMat[r][c] = sum;
        }
    }

    // 释放 mat 原空间，复制 temp 到 mat
    // 这里简单替换指针和尺寸
    // 实际要避免内存泄漏
    // 你可以用 deepCopyOf 函数实现
    for (int r = 0; r < mat->mRows; ++r) {
        free(mat->mMat[r]);
    }
    free(mat->mMat);
    mat->mMat = temp.mMat;
    mat->mRows = temp.mRows;
    mat->mCols = temp.mCols;
    // temp.mMat 指针已经转移给 mat，别 free temp.mMat 了
}

void intnn_self_elem_mul_mat(intnn_mat* mat, const intnn_mat* b) {
    if (!dimsEqual(mat, b))
        assert(0);
    for (int r = 0; r < mat->mRows; ++r) {
        for (int c = 0; c < mat->mCols; ++c) {
            mat->mMat[r][c] *= b->mMat[r][c];
        }
    }
}

void intnn_self_elem_div_mat(intnn_mat* mat, const intnn_mat* b) {
    if (!dimsEqual(mat, b))
        assert(0);
    for (int r = 0; r < mat->mRows; ++r) {
        for (int c = 0; c < mat->mCols; ++c) {
            if (b->mMat[r][c] == 0) {
                mat->mMat[r][c] = INT_MAX;  // 避免除零
            } else {
                mat->mMat[r][c] /= b->mMat[r][c];
            }
        }
    }
}

void intnn_transpose_of(intnn_mat* out, const intnn_mat* in) {
    resetZero(out, in->mCols, in->mRows);
    for (int r = 0; r < in->mRows; ++r) {
        for (int c = 0; c < in->mCols; ++c) {
            out->mMat[c][r] = in->mMat[r][c];
        }
    }
}

void intnn_rotate180_of(intnn_mat* out, const intnn_mat* in) {
    resetZero(out, in->mRows, in->mCols);
    for (int r = 0; r < in->mRows; ++r) {
        for (int c = 0; c < in->mCols; ++c) {
            out->mMat[in->mRows - 1 - r][in->mCols - 1 - c] = in->mMat[r][c];
        }
    }
}

// 简单实现 floor(sqrt(x))，用整数逼近
static int floorSqrt(int x) {
    if (x < 0)
        return 0;
    int res = 0;
    while ((res + 1) * (res + 1) <= x) {
        ++res;
    }
    return res;
}

void intnn_square_root_of(intnn_mat* out, const intnn_mat* in) {
    resetZero(out, in->mRows, in->mCols);
    for (int r = 0; r < in->mRows; ++r) {
        for (int c = 0; c < in->mCols; ++c) {
            out->mMat[r][c] = floorSqrt(in->mMat[r][c]);
        }
    }
}

void intnn_slice_of(intnn_mat* out,
                    const intnn_mat* in,
                    int rowStart,
                    int rowEnd,
                    int colStart,
                    int colEnd) {
    if (rowStart < 0 || colStart < 0 || rowEnd >= in->mRows ||
        colEnd >= in->mCols) {
        printf("[WARN] slice_of: Index out of bounds\n");
        assert(0);
    }

    if (rowEnd < rowStart || colEnd < colStart)
        assert(0);

    int rows = rowEnd - rowStart + 1;
    int cols = colEnd - colStart + 1;
    resetZero(out, rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out->mMat[r][c] = in->mMat[rowStart + r][colStart + c];
        }
    }
}

void intnn_indexed_slice_of(intnn_mat* out,
                            const intnn_mat* in,
                            int* indices,
                            int start,
                            int end) {
    if (start < 0 || end > in->mRows || start >= end)
        assert(0);
    int size = end - start;
    resetZero(out, size, in->mCols);
    for (int r = 0; r < size; ++r) {
        int idx = indices[start + r];
        if (idx < 0 || idx >= in->mRows)
            assert(0);
        for (int c = 0; c < in->mCols; ++c) {
            out->mMat[r][c] = in->mMat[idx][c];
        }
    }
}

void intnn_random_k_samples_of(intnn_mat* out, const intnn_mat* in, int k) {
    if (k > in->mRows)
        assert(0);

    int total = in->mRows;
    int* indices = malloc(sizeof(int) * total);
    for (int i = 0; i < total; ++i)
        indices[i] = i;

    // Fisher-Yates 洗牌
    for (int i = total - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int tmp = indices[j];
        indices[j] = indices[i];
        indices[i] = tmp;
    }

    resetZero(out, k, in->mCols);
    for (int r = 0; r < k; ++r) {
        for (int c = 0; c < in->mCols; ++c) {
            out->mMat[r][c] = in->mMat[indices[r]][c];
        }
    }

    free(indices);
}

void intnn_print_mat(const intnn_mat* mat) {
    for (int r = 0; r < mat->mRows; ++r) {
        for (int c = 0; c < mat->mCols; ++c) {
            printf("%d ", mat->mMat[r][c]);
        }
        printf("\n");
    }
}

void intnn_mat_update_lr(intnn_mat* target,
                         const intnn_mat* update,
                         int lr_inverse) {
    // 先判断维度是否一致
    assert(target->mRows == update->mRows && target->mCols == update->mCols);

    // lr_inverse <= 0 时设为 1，防止除零或负数
    if (lr_inverse <= 0) {
        lr_inverse = 1;
    }

    for (int r = 0; r < target->mRows; ++r) {
        for (int c = 0; c < target->mCols; ++c) {
            // 这里做整数除法，符合原函数
            target->mMat[r][c] += update->mMat[r][c] / lr_inverse;
        }
    }
}

int intnn_count_max_match(const intnn_mat* predictions, const intnn_mat* targets) {
    if (predictions->mRows != targets->mRows) {
        printf("prediction: %d, targets: %d\n", predictions->mRows, targets->mRows);
        assert(0);
    }
    assert(predictions->mRows == targets->mRows);
    assert(predictions->mCols == targets->mCols);

    int count = 0;
    for (int i = 0; i < predictions->mRows; ++i) {
        int maxPredIdx = 0;
        int maxTargetIdx = 0;

        for (int j = 1; j < predictions->mCols; ++j) {
            if (predictions->mMat[i][j] > predictions->mMat[i][maxPredIdx]) {
                maxPredIdx = j;
            }
            if (targets->mMat[i][j] > targets->mMat[i][maxTargetIdx]) {
                maxTargetIdx = j;
            }
        }

        if (maxPredIdx == maxTargetIdx) {
            ++count;
        }
    }

    return count;
}