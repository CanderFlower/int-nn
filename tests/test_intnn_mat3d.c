#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "intnn_mat.h"
#include "intnn_mat3d.h"

#define TEST_ASSERT(cond, msg)        \
    if (!(cond)) {                    \
        printf("[FAILED] %s\n", msg); \
        exit(1);                      \
    } else {                          \
        printf("[PASSED] %s\n", msg); \
    }

void test_create_and_set_get() {
    intnn_mat3d* m3d = intnn_create_mat3d(2, 3, 4);
    TEST_ASSERT(m3d != NULL, "create mat3d");

    intnn_mat3d_set_elem(m3d, 1, 2, 3, 42);
    int val = intnn_mat3d_get_elem(m3d, 1, 2, 3);
    TEST_ASSERT(val == 42, "set and get elem");

    intnn_free_mat3d(m3d);
}

void test_reset_and_dims_equal() {
    intnn_mat3d* m3d1 = intnn_create_mat3d(2, 2, 2);
    intnn_reset_zero3d(m3d1, 2, 2, 2);

    intnn_mat3d* m3d2 = intnn_create_mat3d(2, 2, 2);
    TEST_ASSERT(intnn_mat3d_dims_equal(m3d1, m3d2), "dims equal");
    TEST_ASSERT(intnn_mat3d_dims_equal_size(m3d1, 2, 2, 2), "dims equal size");

    intnn_free_mat3d(m3d1);
    intnn_free_mat3d(m3d2);
}

void test_self_add_and_elem_mul_div() {
    intnn_mat3d* a = intnn_create_mat3d(1, 2, 2);
    intnn_mat3d* b = intnn_create_mat3d(1, 2, 2);

    intnn_mat3d_set_elem(a, 0, 0, 0, 1);
    intnn_mat3d_set_elem(b, 0, 0, 0, 2);
    intnn_mat3d_self_add(a, b);
    TEST_ASSERT(intnn_mat3d_get_elem(a, 0, 0, 0) == 3, "self add");

    intnn_mat3d_set_elem(a, 0, 0, 1, 4);
    intnn_mat3d_set_elem(b, 0, 0, 1, 2);
    intnn_mat3d_self_elem_mul(a, b);
    TEST_ASSERT(intnn_mat3d_get_elem(a, 0, 0, 1) == 8, "self elem mul");

    intnn_mat3d_self_div_const(a, 2);
    TEST_ASSERT(intnn_mat3d_get_elem(a, 0, 0, 0) == 1, "self div const");

    intnn_mat3d_set_elem(b, 0, 0, 0, 1);
    intnn_mat3d_self_elem_div(a, b);
    TEST_ASSERT(intnn_mat3d_get_elem(a, 0, 0, 0) == 1, "self elem div");

    intnn_free_mat3d(a);
    intnn_free_mat3d(b);
}

void test_rotate180() {
    intnn_mat3d* a = intnn_create_mat3d(1, 2, 2);
    intnn_mat3d_set_elem(a, 0, 0, 0, 1);
    intnn_mat3d_set_elem(a, 0, 0, 1, 2);
    intnn_mat3d_set_elem(a, 0, 1, 0, 3);
    intnn_mat3d_set_elem(a, 0, 1, 1, 4);

    intnn_mat3d* out = intnn_create_mat3d(1, 2, 2);
    intnn_mat3d_rotate180(out, a);

    TEST_ASSERT(intnn_mat3d_get_elem(out, 0, 0, 0) == 4, "rotate180[0][0][0] == 4");
    TEST_ASSERT(intnn_mat3d_get_elem(out, 0, 0, 1) == 3, "rotate180[0][0][1] == 3");
    TEST_ASSERT(intnn_mat3d_get_elem(out, 0, 1, 0) == 2, "rotate180[0][1][0] == 2");
    TEST_ASSERT(intnn_mat3d_get_elem(out, 0, 1, 1) == 1, "rotate180[0][1][1] == 1");

    intnn_free_mat3d(a);
    intnn_free_mat3d(out);
}

void test_make_from_mat() {
    intnn_mat* mat = intnn_create_mat(1, 8);
    for (int i = 0; i < 8; ++i) {
        intnn_set_elem(mat, 0, i, i + 1);
    }

    intnn_mat3d* m3d = intnn_create_mat3d(2, 2, 2);
    intnn_mat3d_make_from_mat(m3d, 2, 2, 2, mat);

    TEST_ASSERT(intnn_mat3d_get_elem(m3d, 0, 0, 0) == 1, "make from mat [0][0][0] == 1");
    TEST_ASSERT(intnn_mat3d_get_elem(m3d, 1, 1, 1) == 8, "make from mat [1][1][1] == 8");

    intnn_free_mat(mat);
    intnn_free_mat3d(m3d);
}

void test_deep_copy() {
    intnn_mat3d* a = intnn_create_mat3d(1, 2, 2);
    intnn_mat3d_set_elem(a, 0, 1, 1, 77);

    intnn_mat3d* b = intnn_create_mat3d(1, 2, 2);
    intnn_mat3d_deep_copy(b, a);

    TEST_ASSERT(intnn_mat3d_get_elem(b, 0, 1, 1) == 77, "deep copy value");
    intnn_mat3d_set_elem(a, 0, 1, 1, 88);
    TEST_ASSERT(intnn_mat3d_get_elem(b, 0, 1, 1) == 77, "deep copy independence");

    intnn_free_mat3d(a);
    intnn_free_mat3d(b);
}

int main() {
    test_create_and_set_get();
    test_reset_and_dims_equal();
    test_self_add_and_elem_mul_div();
    test_rotate180();
    test_make_from_mat();
    test_deep_copy();
    printf("All mat3d tests passed.\n");
    return 0;
}
