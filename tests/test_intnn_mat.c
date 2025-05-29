#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "intnn_mat.h"

// 简单辅助断言打印
#define TEST_ASSERT(cond, msg)        \
    if (!(cond)) {                    \
        printf("[FAILED] %s\n", msg); \
        exit(1);                      \
    } else {                          \
        printf("[PASSED] %s\n", msg); \
    }

// 检查矩阵所有元素等于某值
void check_all_equal(intnn_mat* mat, int val, const char* msg) {
    for (int r = 0; r < intnn_rows(mat); r++) {
        for (int c = 0; c < intnn_cols(mat); c++) {
            TEST_ASSERT(intnn_get_elem(mat, r, c) == val, msg);
        }
    }
}

void test_create_and_free() {
    intnn_mat* m = intnn_create_mat(3, 4);
    TEST_ASSERT(m != NULL, "Create matrix");
    TEST_ASSERT(intnn_rows(m) == 3 && intnn_cols(m) == 4, "Matrix dimension");
    intnn_free_mat(m);
}

void test_set_and_get_elem() {
    intnn_mat* m = intnn_create_mat(2, 2);
    intnn_set_elem(m, 0, 0, 10);
    intnn_set_elem(m, 0, 1, 20);
    intnn_set_elem(m, 1, 0, 30);
    intnn_set_elem(m, 1, 1, 40);
    TEST_ASSERT(intnn_get_elem(m, 0, 0) == 10, "Get elem 0,0");
    TEST_ASSERT(intnn_get_elem(m, 1, 1) == 40, "Get elem 1,1");
    intnn_free_mat(m);
}

void test_set_all_constant() {
    intnn_mat* m = intnn_create_mat(4, 3);
    intnn_set_all_constant(m, 7);
    check_all_equal(m, 7, "Set all constant to 7");
    intnn_free_mat(m);
}

void test_copy_mat() {
    intnn_mat* src = intnn_create_mat(2, 3);
    intnn_set_all_constant(src, 5);
    intnn_set_elem(src, 1, 2, 10);
    intnn_mat* copy = intnn_copy_mat(src);
    TEST_ASSERT(copy != NULL, "Copy matrix not NULL");
    TEST_ASSERT(intnn_dims_equal(src, copy), "Copied dims equal");
    TEST_ASSERT(intnn_get_elem(copy, 1, 2) == 10, "Copied elem value");
    intnn_free_mat(src);
    intnn_free_mat(copy);
}

void test_sum_and_num_elems() {
    intnn_mat* m = intnn_create_mat(3, 3);
    intnn_set_all_constant(m, 1);
    TEST_ASSERT(intnn_sum(m) == 9, "Sum equals 9");
    TEST_ASSERT(intnn_num_elems(m) == 9, "Num elems equals 9");
    intnn_free_mat(m);
}

void test_dims_equal_and_dims_equal_size() {
    intnn_mat* m1 = intnn_create_mat(2, 3);
    intnn_mat* m2 = intnn_create_mat(2, 3);
    intnn_mat* m3 = intnn_create_mat(3, 2);
    TEST_ASSERT(intnn_dims_equal(m1, m2), "Dims equal for same size");
    TEST_ASSERT(!intnn_dims_equal(m1, m3), "Dims not equal for different size");
    TEST_ASSERT(intnn_dims_equal_size(m1, 2, 3), "Dims equal size match");
    TEST_ASSERT(!intnn_dims_equal_size(m1, 3, 2),
                "Dims not equal size mismatch");
    intnn_free_mat(m1);
    intnn_free_mat(m2);
    intnn_free_mat(m3);
}

void test_get_max_index_and_min_max() {
    intnn_mat* m = intnn_create_mat(2, 3);
    int data[] = {1, 5, 3, 4, 2, 6};
    for (int i = 0; i < 6; i++) {
        intnn_set_elem(m, i / 3, i % 3, data[i]);
    }
    TEST_ASSERT(intnn_get_max_index_in_row(m, 0) == 1, "Max index in row 0");
    TEST_ASSERT(intnn_get_row_min(m, 1) == 2, "Row 1 min");
    TEST_ASSERT(intnn_get_row_max(m, 1) == 6, "Row 1 max");
    TEST_ASSERT(intnn_get_col_min(m, 1) == 2, "Col 1 min");
    TEST_ASSERT(intnn_get_col_max(m, 2) == 6, "Col 2 max");
    intnn_free_mat(m);
}

void test_average_variance_stdev() {
    intnn_mat* m = intnn_create_mat(1, 5);
    int data[] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5; i++) {
        intnn_set_elem(m, 0, i, data[i]);
    }
    int avg = intnn_average(m);
    TEST_ASSERT(avg == 3, "Average calculation");
    int var = intnn_variance_with_avg(m, avg);
    TEST_ASSERT(var == 2, "Variance calculation");
    int var2 = intnn_variance(m);
    TEST_ASSERT(var2 == 2, "Variance without avg");
    int stdev = intnn_stdev_with_avg(m, avg);
    TEST_ASSERT(stdev == 1, "Stdev calculation");
    int stdev2 = intnn_stdev(m);
    TEST_ASSERT(stdev2 == 1, "Stdev without avg");
    intnn_free_mat(m);
}

void test_inplace_operations() {
    intnn_mat* m = intnn_create_mat(2, 2);
    intnn_set_all_constant(m, 2);
    intnn_self_add_const(m, 3);
    check_all_equal(m, 5, "Self add const 3");
    intnn_self_mul_const(m, 2);
    check_all_equal(m, 10, "Self mul const 2");
    intnn_self_div_const(m, 5);
    check_all_equal(m, 2, "Self div const 5");
    intnn_self_elem_add_const(m, 0, 0, 10);
    TEST_ASSERT(intnn_get_elem(m, 0, 0) == 12, "Elem add const at (0,0)");
    intnn_free_mat(m);
}

void test_out_of_place_mat_operations() {
    intnn_mat* a = intnn_create_mat(2, 2);
    intnn_mat* b = intnn_create_mat(2, 2);
    intnn_set_all_constant(a, 2);
    intnn_set_all_constant(b, 3);
    intnn_mat* out = intnn_create_mat(2, 2);

    intnn_mat_add_mat(out, a, b);
    TEST_ASSERT(intnn_get_elem(out, 0, 0) == 5, "Mat add");

    intnn_mat_elem_mul_mat(out, a, b);
    TEST_ASSERT(intnn_get_elem(out, 1, 1) == 6, "Elem mul");

    intnn_mat_mul_const(out, a, 4);
    TEST_ASSERT(intnn_get_elem(out, 0, 1) == 8, "Mul const");

    intnn_mat_add_const(out, b, 10);
    TEST_ASSERT(intnn_get_elem(out, 1, 0) == 13, "Add const");

    intnn_mat_div_const(out, out, 2);
    TEST_ASSERT(intnn_get_elem(out, 0, 0) == 6, "Div const");

    // 测试矩阵乘法 a * b (2x2 * 2x2)
    intnn_set_elem(a, 0, 0, 1);
    intnn_set_elem(a, 0, 1, 2);
    intnn_set_elem(a, 1, 0, 3);
    intnn_set_elem(a, 1, 1, 4);

    intnn_set_elem(b, 0, 0, 5);
    intnn_set_elem(b, 0, 1, 6);
    intnn_set_elem(b, 1, 0, 7);
    intnn_set_elem(b, 1, 1, 8);

    intnn_mat_mul_mat(out, a, b);
    TEST_ASSERT(intnn_get_elem(out, 0, 0) == 19, "Mat mul mat (0,0)");
    TEST_ASSERT(intnn_get_elem(out, 1, 1) == 50, "Mat mul mat (1,1)");

    intnn_free_mat(a);
    intnn_free_mat(b);
    intnn_free_mat(out);
}

void test_transforms() {
    intnn_mat* m = intnn_create_mat(2, 2);
    int data[] = {1, 2, 3, 4};
    for (int i = 0; i < 4; i++) {
        intnn_set_elem(m, i / 2, i % 2, data[i]);
    }
    intnn_mat* out = intnn_create_mat(2, 2);

    intnn_transpose_of(out, m);
    TEST_ASSERT(intnn_get_elem(out, 0, 1) == 3, "Transpose check");

    intnn_rotate180_of(out, m);
    TEST_ASSERT(intnn_get_elem(out, 0, 0) == 4, "Rotate180 check");

    intnn_mat* slice = intnn_create_mat(2, 1);
    intnn_slice_of(slice, m, 0, 1, 1, 1);
    TEST_ASSERT(intnn_get_elem(slice, 0, 0) == 2, "Slice_of check (0,0)");
    TEST_ASSERT(intnn_get_elem(slice, 1, 0) == 4, "Slice_of check (1,0)");

    intnn_free_mat(m);
    intnn_free_mat(out);
    intnn_free_mat(slice);
}

int main() {
    test_create_and_free();
    test_set_and_get_elem();
    test_set_all_constant();
    test_copy_mat();
    test_sum_and_num_elems();
    test_dims_equal_and_dims_equal_size();
    test_get_max_index_and_min_max();
    test_average_variance_stdev();
    test_inplace_operations();
    test_out_of_place_mat_operations();
    test_transforms();

    printf("All tests passed!\n");
    return 0;
}
