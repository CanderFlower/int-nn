#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "intnn_mat.h"
#include "intnn_loss.h"

#define TEST_ASSERT(cond, msg)        \
    if (!(cond)) {                    \
        printf("[FAILED] %s\n", msg); \
        exit(1);                      \
    } else {                          \
        printf("[PASSED] %s\n", msg); \
    }

void test_scalar_l2() {
    int y = 10;
    int y_hat = 8;
    int loss = intnn_scalar_l2_loss(y, y_hat);
    int delta = intnn_scalar_l2_loss_delta(y, y_hat);
    TEST_ASSERT(loss == (2 * 2) / 2, "scalar L2 loss = (10-8)^2/2");
    TEST_ASSERT(delta == -2, "scalar L2 loss delta = (8-10)");
}

void test_batch_l2() {
    intnn_mat* y = intnn_create_mat(2, 2);
    intnn_mat* y_hat = intnn_create_mat(2, 2);
    intnn_mat* loss = intnn_create_mat(2, 1);
    intnn_mat* delta = intnn_create_mat(2, 2);

    intnn_set_all_constant(y, 0);
    intnn_set_all_constant(y_hat, 0);
    intnn_set_all_constant(loss, 0);
    intnn_set_all_constant(delta, 0);

    intnn_set_elem(y, 0, 0, 10);
    intnn_set_elem(y_hat, 0, 0, 7); // (10-7)^2/2 = 9/2

    int total_loss = intnn_batch_l2_loss(loss, y, y_hat);
    TEST_ASSERT(intnn_get_elem(loss, 0, 0) == 9 / 2, "batch L2 loss [0]");
    TEST_ASSERT(total_loss == 9 / 2, "total batch L2 loss");

    intnn_batch_l2_loss_delta(delta, y, y_hat);
    TEST_ASSERT(intnn_get_elem(delta, 0, 0) == -3, "batch L2 delta [0][0]");

    intnn_free_mat(y);
    intnn_free_mat(y_hat);
    intnn_free_mat(loss);
    intnn_free_mat(delta);
}

void test_vector_pocket_cross_loss() {
    intnn_mat* y = intnn_create_mat(1, 3);
    intnn_mat* y_hat = intnn_create_mat(1, 3);
    intnn_mat* loss = intnn_create_mat(1, 1);

    intnn_set_all_constant(y, 0);
    intnn_set_all_constant(y_hat, 0);
    intnn_set_all_constant(loss, 0);

    intnn_set_elem(y, 0, 1, INT_MAX);
    intnn_set_elem(y_hat, 0, 1, 100);

    int sum = intnn_vector_pocket_cross_loss(loss, y, y_hat);
    int expected = INT_MAX - 100;

    TEST_ASSERT(intnn_get_elem(loss, 0, 0) == expected, "vector pocket cross loss value");
    TEST_ASSERT(sum == expected, "vector pocket cross loss sum");

    intnn_free_mat(y);
    intnn_free_mat(y_hat);
    intnn_free_mat(loss);
}

void test_batch_pocket_cross_loss_and_delta() {
    intnn_mat* y = intnn_create_mat(2, 2);
    intnn_mat* y_hat = intnn_create_mat(2, 2);
    intnn_mat* loss = intnn_create_mat(2, 1);
    intnn_mat* delta = intnn_create_mat(2, 2);

    intnn_set_all_constant(y, 0);
    intnn_set_all_constant(y_hat, 0);
    intnn_set_all_constant(loss, 0);
    intnn_set_all_constant(delta, 0);

    intnn_set_elem(y, 0, 0, INT_MAX);
    intnn_set_elem(y_hat, 0, 0, 123);
    int expected_loss = INT_MAX - 123;

    int loss_sum = intnn_batch_pocket_cross_loss(loss, y, y_hat);
    TEST_ASSERT(loss_sum == expected_loss, "batch pocket cross loss sum");
    TEST_ASSERT(intnn_get_elem(loss, 0, 0) == expected_loss, "loss[0] check");

    int delta_sum = intnn_batch_pocket_cross_loss_delta(delta, y, y_hat);
    TEST_ASSERT(delta_sum == -1, "pocket cross delta sum");
    TEST_ASSERT(intnn_get_elem(delta, 0, 0) == -1, "delta[0][0] check");

    intnn_free_mat(y);
    intnn_free_mat(y_hat);
    intnn_free_mat(loss);
    intnn_free_mat(delta);
}

void test_batch_cross_entropy_loss_and_delta() {
    intnn_mat* y = intnn_create_mat(1, 2);
    intnn_mat* y_hat = intnn_create_mat(1, 2);
    intnn_mat* loss = intnn_create_mat(1, 1);
    intnn_mat* delta = intnn_create_mat(1, 2);

    intnn_set_all_constant(y, 0);
    intnn_set_all_constant(y_hat, 0);
    intnn_set_all_constant(loss, 0);
    intnn_set_all_constant(delta, 0);

    intnn_set_elem(y, 0, 1, 1);
    intnn_set_elem(y_hat, 0, 1, INTNN_MAX - 3); // delta = -3, loss = 9/2

    int loss_sum = intnn_batch_cross_entropy_loss(loss, y, y_hat);
    TEST_ASSERT(loss_sum == (3 * 3) / 2, "cross entropy loss sum");
    TEST_ASSERT(intnn_get_elem(loss, 0, 0) == (3 * 3) / 2, "cross entropy loss value");

    int delta_sum = intnn_batch_cross_entropy_loss_delta(delta, y, y_hat);
    TEST_ASSERT(delta_sum == -3, "cross entropy delta sum");
    TEST_ASSERT(intnn_get_elem(delta, 0, 1) == -3, "cross entropy delta value");

    intnn_free_mat(y);
    intnn_free_mat(y_hat);
    intnn_free_mat(loss);
    intnn_free_mat(delta);
}

int main() {
    printf("Running intnn_loss tests...\n");

    test_scalar_l2();
    test_batch_l2();
    test_vector_pocket_cross_loss();
    test_batch_pocket_cross_loss_and_delta();
    test_batch_cross_entropy_loss_and_delta();

    printf("All tests passed!\n");
    return 0;
}
