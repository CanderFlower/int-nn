#include "intnn_mat3d.h"
#include <stdlib.h>
#include <assert.h>

intnn_mat3d* intnn_create_mat3d(int depth, int rows, int cols) {
    intnn_mat3d* mat3d = (intnn_mat3d*)malloc(sizeof(intnn_mat3d));
    if (!mat3d) return NULL;

    mat3d->mDepth = (depth > 0) ? depth : 0;
    mat3d->mRows = (rows > 0) ? rows : 0;
    mat3d->mCols = (cols > 0) ? cols : 0;
    mat3d->mDeleteOnDestruct = true;

    if (mat3d->mDepth == 0 || mat3d->mRows == 0 || mat3d->mCols == 0) {
        mat3d->mMat3d = NULL;
        return mat3d;
    }

    mat3d->mMat3d = (intnn_mat**)malloc(sizeof(intnn_mat*) * mat3d->mDepth);
    if (!mat3d->mMat3d) {
        free(mat3d);
        return NULL;
    }

    for (int i = 0; i < mat3d->mDepth; i++) {
        mat3d->mMat3d[i] = intnn_create_mat(mat3d->mRows, mat3d->mCols);
        if (!mat3d->mMat3d[i]) {
            // 失败时释放已分配内存
            for (int j = 0; j < i; j++) {
                intnn_free_mat(mat3d->mMat3d[j]);
            }
            free(mat3d->mMat3d);
            free(mat3d);
            return NULL;
        }
    }

    return mat3d;
}

void intnn_free_mat3d(intnn_mat3d* mat3d) {
    if (!mat3d) return;
    if (mat3d->mDeleteOnDestruct && mat3d->mMat3d) {
        for (int i = 0; i < mat3d->mDepth; i++) {
            if (mat3d->mMat3d[i]) {
                intnn_free_mat(mat3d->mMat3d[i]);
                mat3d->mMat3d[i] = NULL;
            }
        }
        free(mat3d->mMat3d);
        mat3d->mMat3d = NULL;
    }
    free(mat3d);
}

void intnn_reset_zero3d(intnn_mat3d* mat3d, int depth, int rows, int cols) {
    if (!mat3d) return;

    if (mat3d->mMat3d) {
        // 先释放原来的矩阵数据
        for (int i = 0; i < mat3d->mDepth; i++) {
            if (mat3d->mMat3d[i]) {
                intnn_free_mat(mat3d->mMat3d[i]);
            }
        }
        free(mat3d->mMat3d);
    }

    mat3d->mDepth = (depth > 0) ? depth : 0;
    mat3d->mRows = (rows > 0) ? rows : 0;
    mat3d->mCols = (cols > 0) ? cols : 0;

    if (mat3d->mDepth == 0 || mat3d->mRows == 0 || mat3d->mCols == 0) {
        mat3d->mMat3d = NULL;
        return;
    }

    mat3d->mMat3d = (intnn_mat**)malloc(sizeof(intnn_mat*) * mat3d->mDepth);
    if (!mat3d->mMat3d) return; // 分配失败不做事

    for (int i = 0; i < mat3d->mDepth; i++) {
        mat3d->mMat3d[i] = intnn_create_mat(mat3d->mRows, mat3d->mCols);
        if (!mat3d->mMat3d[i]) {
            // 失败时释放已分配
            for (int j = 0; j < i; j++) {
                intnn_free_mat(mat3d->mMat3d[j]);
            }
            free(mat3d->mMat3d);
            mat3d->mMat3d = NULL;
            mat3d->mDepth = 0;
            mat3d->mRows = 0;
            mat3d->mCols = 0;
            return;
        }
    }
}

int intnn_mat3d_rows(const intnn_mat3d* mat3d) {
    assert(mat3d);
    return mat3d->mRows;
}

int intnn_mat3d_cols(const intnn_mat3d* mat3d) {
    assert(mat3d);
    return mat3d->mCols;
}

int intnn_mat3d_depth(const intnn_mat3d* mat3d) {
    assert(mat3d);
    return mat3d->mDepth;
}

int intnn_mat3d_get_elem(const intnn_mat3d* mat3d, int d, int r, int c) {
    assert(mat3d);
    assert(d >= 0 && d < mat3d->mDepth);
    assert(r >= 0 && r < mat3d->mRows);
    assert(c >= 0 && c < mat3d->mCols);
    return intnn_get_elem(mat3d->mMat3d[d], r, c);
}

void intnn_mat3d_set_elem(intnn_mat3d* mat3d, int d, int r, int c, int val) {
    assert(mat3d);
    assert(d >= 0 && d < mat3d->mDepth);
    assert(r >= 0 && r < mat3d->mRows);
    assert(c >= 0 && c < mat3d->mCols);
    intnn_set_elem(mat3d->mMat3d[d], r, c, val);
}

intnn_mat* intnn_mat3d_get_mat_at_depth(intnn_mat3d* mat3d, int d) {
    assert(mat3d);
    assert(d >= 0 && d < mat3d->mDepth);
    return mat3d->mMat3d[d];
}

bool intnn_mat3d_dims_equal(const intnn_mat3d* m1, const intnn_mat3d* m2) {
    assert(m1 && m2);
    return (m1->mDepth == m2->mDepth) &&
           (m1->mRows == m2->mRows) &&
           (m1->mCols == m2->mCols);
}

bool intnn_mat3d_dims_equal_size(const intnn_mat3d* mat3d, int depth, int rows, int cols) {
    assert(mat3d);
    return (mat3d->mDepth == depth) &&
           (mat3d->mRows == rows) &&
           (mat3d->mCols == cols);
}

// 下面开始写运算接口
void intnn_mat3d_normalize_minmax(intnn_mat3d* mat3d, int newMin, int newMax) {
    assert(mat3d);
    for (int d = 0; d < mat3d->mDepth; d++) {
        intnn_normalize_minmax(mat3d->mMat3d[d], newMin, newMax);
    }
}

void intnn_mat3d_add(intnn_mat3d* out, const intnn_mat3d* a, const intnn_mat3d* b) {
    assert(out && a && b);
    assert(intnn_mat3d_dims_equal(a, b));

    // 如果 out 维度不匹配则重置
    if (!intnn_mat3d_dims_equal(out, a)) {
        intnn_reset_zero3d(out, a->mDepth, a->mRows, a->mCols);
    }

    for (int d = 0; d < out->mDepth; d++) {
        intnn_mat_add_mat(out->mMat3d[d], a->mMat3d[d], b->mMat3d[d]);
    }
}

void intnn_mat3d_elem_div(intnn_mat3d* out, const intnn_mat3d* a, const intnn_mat3d* b) {
    assert(out && a && b);
    assert(intnn_mat3d_dims_equal(a, b));

    if (!intnn_mat3d_dims_equal(out, a)) {
        intnn_reset_zero3d(out, a->mDepth, a->mRows, a->mCols);
    }

    for (int d = 0; d < out->mDepth; d++) {
        intnn_mat_elem_div_mat(out->mMat3d[d], a->mMat3d[d], b->mMat3d[d]);
    }
}

void intnn_mat3d_self_add(intnn_mat3d* self, const intnn_mat3d* other) {
    if (!intnn_mat3d_dims_equal(self, other)) return;

    for (int d = 0; d < self->mDepth; ++d) {
        intnn_self_add_mat(self->mMat3d[d], other->mMat3d[d]);
    }
}
void intnn_mat3d_self_div_const(intnn_mat3d* self, int val) {
    for (int d = 0; d < self->mDepth; ++d) {
        intnn_self_div_const(self->mMat3d[d], val);
    }
}
void intnn_mat3d_self_elem_mul(intnn_mat3d* self, const intnn_mat3d* other) {
    if (!intnn_mat3d_dims_equal(self, other)) return;

    for (int d = 0; d < self->mDepth; ++d) {
        intnn_self_elem_mul_mat(self->mMat3d[d], other->mMat3d[d]);
    }
}
void intnn_mat3d_self_elem_div(intnn_mat3d* self, const intnn_mat3d* other) {
    if (!intnn_mat3d_dims_equal(self, other)) return;

    for (int d = 0; d < self->mDepth; ++d) {
        intnn_self_elem_div_mat(self->mMat3d[d], other->mMat3d[d]);
    }
}
void intnn_mat3d_rotate180(intnn_mat3d* out, const intnn_mat3d* in) {
    intnn_reset_zero3d(out, in->mDepth, in->mRows, in->mCols);

    for (int d = 0; d < in->mDepth; ++d) {
        intnn_rotate180_of(out->mMat3d[d], in->mMat3d[d]);
    }
}
void intnn_mat3d_make_from_mat(intnn_mat3d* out, int depth, int rows, int cols, const intnn_mat* mat) {
    if (depth * rows * cols != intnn_num_elems(mat)) return;

    intnn_reset_zero3d(out, depth, rows, cols);

    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                int idx = d * (rows * cols) + r * cols + c;
                int val = intnn_get_elem(mat, idx / mat->mCols, idx % mat->mCols);
                intnn_mat3d_set_elem(out, d, r, c, val);
            }
        }
    }
}
void intnn_mat3d_deep_copy(intnn_mat3d* out, const intnn_mat3d* in) {
    intnn_reset_zero3d(out, in->mDepth, in->mRows, in->mCols);

    for (int d = 0; d < in->mDepth; ++d) {
        intnn_mat* copy = intnn_copy_mat(in->mMat3d[d]);
        intnn_free_mat(out->mMat3d[d]);
        out->mMat3d[d] = copy;
    }
}
void intnn_mat3d_print(const intnn_mat3d* mat3d) {
    printf("Matrix3D: depth=%d, rows=%d, cols=%d\n", mat3d->mDepth, mat3d->mRows, mat3d->mCols);
    for (int d = 0; d < mat3d->mDepth; ++d) {
        printf("Depth %d:\n", d);
        intnn_print_mat(mat3d->mMat3d[d]);
        printf("\n");
    }
}
