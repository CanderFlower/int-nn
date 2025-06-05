#ifndef INTNN_MAT_H
#define INTNN_MAT_H

#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

#include "intnn_tools.h"
#include "intnn_consts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int mRows;
    int mCols;
    int** mMat;
    bool mDeleteOnDestruct;
    const char* mName;
} intnn_mat;

// 构造与释放
intnn_mat* intnn_create_mat(int rows, int cols);
void intnn_free_mat(intnn_mat* mat);
intnn_mat* intnn_copy_mat(const intnn_mat* src);
void intnn_set_delete_on_destruct(intnn_mat* mat, bool flag);

// 数据初始化与设置
void intnn_set_all_constant(intnn_mat* mat, int value);
void intnn_set_random(intnn_mat* mat, bool allowZero, int minVal, int maxVal);
void intnn_set_elem(intnn_mat* mat, int r, int c, int val);
intnn_mat* intnn_set_mat_from_array(int rows, int cols, int* data);
void intnn_reset_zero(intnn_mat* mat, int rows, int cols);
void intnn_reset_all_ones(intnn_mat* mat, int rows, int cols);

// 获取信息
int intnn_rows(const intnn_mat* mat);
int intnn_cols(const intnn_mat* mat);
int intnn_sum(const intnn_mat* mat);
int intnn_num_elems(const intnn_mat* mat);
int intnn_get_elem(const intnn_mat* mat, int r, int c);
int** intnn_get_data(intnn_mat* mat);
bool intnn_dims_equal(const intnn_mat* m1, const intnn_mat* m2);
bool intnn_dims_equal_size(const intnn_mat* mat, int r, int c);
int intnn_get_max_index_in_row(const intnn_mat* mat, int r);
int intnn_get_row_min(const intnn_mat* mat, int r);
int intnn_get_row_max(const intnn_mat* mat, int r);
int intnn_get_col_min(const intnn_mat* mat, int c);
int intnn_get_col_max(const intnn_mat* mat, int c);

// 数值处理
int intnn_average(const intnn_mat* mat);
int intnn_variance(const intnn_mat* mat);
int intnn_variance_with_avg(const intnn_mat* mat, int avg);
int intnn_stdev(const intnn_mat* mat);
int intnn_stdev_with_avg(const intnn_mat* mat, int avg);

// 标准化/归一化
void intnn_average_colwise(intnn_mat* mat);
void intnn_standardize(intnn_mat* mat, int numSigma, int low, int high);
void intnn_normalize_rowwise(intnn_mat* mat, int newMin, int newMax);
void intnn_normalize_colwise(intnn_mat* mat, int newMin, int newMax);
void intnn_normalize_minmax(intnn_mat* mat, int newMin, int newMax);
void intnn_clamp_mat(intnn_mat* mat, int low, int high);

// 运算接口（in-place 和生成）
void intnn_mat_mul_mat(intnn_mat* out, const intnn_mat* a, const intnn_mat* b);
void intnn_mat_add_mat(intnn_mat* out, const intnn_mat* a, const intnn_mat* b);
void intnn_mat_elem_mul_mat(intnn_mat* out, const intnn_mat* a, const intnn_mat* b);
void intnn_mat_elem_div_mat(intnn_mat* out, const intnn_mat* a, const intnn_mat* b);
void intnn_mat_add_const(intnn_mat* out, const intnn_mat* a, int val);
void intnn_mat_mul_const(intnn_mat* out, const intnn_mat* a, int val);
void intnn_mat_div_const(intnn_mat* out, const intnn_mat* a, int val);

// 自身修改
void intnn_self_add_const(intnn_mat* mat, int val);
void intnn_self_mul_const(intnn_mat* mat, int val);
void intnn_self_div_const(intnn_mat* mat, int val);
void intnn_self_elem_add_const(intnn_mat* mat, int r, int c, int val);
void intnn_self_add_mat(intnn_mat* mat, const intnn_mat* b);
void intnn_self_sub_mat(intnn_mat* mat, const intnn_mat* b);
void intnn_self_mul_mat(intnn_mat* mat, const intnn_mat* b);
void intnn_self_elem_mul_mat(intnn_mat* mat, const intnn_mat* b);
void intnn_self_elem_div_mat(intnn_mat* mat, const intnn_mat* b);

// 变换
void intnn_transpose_of(intnn_mat* out, const intnn_mat* in);
void intnn_rotate180_of(intnn_mat* out, const intnn_mat* in);
void intnn_square_root_of(intnn_mat* out, const intnn_mat* in);
void intnn_slice_of(intnn_mat* out, const intnn_mat* in, int rowStart, int rowEnd, int colStart, int colEnd);
void intnn_indexed_slice_of(intnn_mat* out, const intnn_mat* in, int* indices, int start, int end);
void intnn_random_k_samples_of(intnn_mat* out, const intnn_mat* in, int k);

// 其他
void intnn_print_mat(const intnn_mat* mat);
int intnn_count_max_match(const intnn_mat* predictions, const intnn_mat* targets);

// 学习率更新（类似 SGD）
void intnn_mat_update_lr(intnn_mat* target, const intnn_mat* update, int lr_inverse);

#ifdef __cplusplus
}
#endif

#endif // INTNN_MAT_H
