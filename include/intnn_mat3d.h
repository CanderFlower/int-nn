#ifndef INTNN_MAT3D_H
#define INTNN_MAT3D_H

#include <stdbool.h>
#include "intnn_mat.h"

typedef struct {
    int mDepth;
    int mRows;
    int mCols;
    intnn_mat** mMat3d; // 长度为 mDepth 的指针数组，指向每层矩阵
    bool mDeleteOnDestruct;
} intnn_mat3d;

// 构造与析构
intnn_mat3d* intnn_create_mat3d(int depth, int rows, int cols);
void intnn_free_mat3d(intnn_mat3d* mat3d);
void intnn_reset_zero3d(intnn_mat3d* mat3d, int depth, int rows, int cols);

// 访问维度
int intnn_mat3d_rows(const intnn_mat3d* mat3d);
int intnn_mat3d_cols(const intnn_mat3d* mat3d);
int intnn_mat3d_depth(const intnn_mat3d* mat3d);

// 访问元素
int intnn_mat3d_get_elem(const intnn_mat3d* mat3d, int d, int r, int c);
void intnn_mat3d_set_elem(intnn_mat3d* mat3d, int d, int r, int c, int val);

// 访问某层矩阵指针
intnn_mat* intnn_mat3d_get_mat_at_depth(intnn_mat3d* mat3d, int d);

// 比较维度
bool intnn_mat3d_dims_equal(const intnn_mat3d* m1, const intnn_mat3d* m2);
bool intnn_mat3d_dims_equal_size(const intnn_mat3d* mat3d, int depth, int rows, int cols);

// 运算
void intnn_mat3d_normalize_minmax(intnn_mat3d* mat3d, int newMin, int newMax);
void intnn_mat3d_add(intnn_mat3d* out, const intnn_mat3d* a, const intnn_mat3d* b);
void intnn_mat3d_elem_div(intnn_mat3d* out, const intnn_mat3d* a, const intnn_mat3d* b);
void intnn_mat3d_self_add(intnn_mat3d* self, const intnn_mat3d* other);
void intnn_mat3d_self_div_const(intnn_mat3d* self, int val);
void intnn_mat3d_self_elem_mul(intnn_mat3d* self, const intnn_mat3d* other);
void intnn_mat3d_self_elem_div(intnn_mat3d* self, const intnn_mat3d* other);
void intnn_mat3d_rotate180(intnn_mat3d* out, const intnn_mat3d* in);
void intnn_mat3d_make_from_mat(intnn_mat3d* out, int depth, int rows, int cols, const intnn_mat* mat);
void intnn_mat3d_deep_copy(intnn_mat3d* out, const intnn_mat3d* in);

// 打印
void intnn_mat3d_print(const intnn_mat3d* mat3d);

#endif
