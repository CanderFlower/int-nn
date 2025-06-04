#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <string.h>
#include "intnn_fc_layer.h"
#include "intnn_mat.h"
#include "intnn_actv.h"

#define TEST_ASSERT(cond, msg)        \
    if (!(cond)) {                    \
        printf("[FAILED] %s\n", msg); \
        exit(1);                      \
    } else {                          \
        printf("[PASSED] %s\n", msg); \
    }

// 检查矩阵所有元素与预期整数相等
void check_mat_equal(intnn_mat* mat, int expected[], int len, const char* msg) {
    TEST_ASSERT(mat != NULL, "Matrix is NULL in check_mat_equal");
    int rows = intnn_rows(mat);
    int cols = intnn_cols(mat);
    TEST_ASSERT(rows > 0 && cols > 0, "Matrix dimensions are invalid in check_mat_equal");
    TEST_ASSERT(rows * cols == len, "Matrix size mismatch in check_mat_equal");
    intnn_print_mat(mat);
    for (int i = 0; i < len; i++) {
        int row = i / cols;
        int col = i % cols;
        TEST_ASSERT(row < rows && col < cols, "Matrix index out of bounds in check_mat_equal");
        int val = intnn_get_elem(mat, row, col);
        int expv = expected[i];
        TEST_ASSERT(val == expv, msg);
    }
    printf("[PASSED] %s (all elements match)\n", msg);
}

void test_create_and_free() {
    printf("=== test_create_and_free ===\n");
    intnn_fc_layer* layer = intnn_fc_create(3, 4);
    TEST_ASSERT(layer != NULL, "Layer created");
    TEST_ASSERT(layer->mInDim == 3 && layer->mOutDim == 4, "Dimensions correct");
    TEST_ASSERT(layer->mWeight != NULL && layer->mBias != NULL, "Weight and Bias allocated");
    intnn_fc_free(layer);
    free(layer);
}

void test_set_and_get_name() {
    printf("=== test_set_and_get_name ===\n");
    intnn_fc_layer* layer = intnn_fc_create(2, 2);
    intnn_fc_set_name(layer, "MyLayer");
    TEST_ASSERT(strcmp(layer->mName, "MyLayer") == 0, "Layer name set/get");
    intnn_fc_free(layer);
    free(layer);
}

void test_random_and_he_initialization() {
    printf("=== test_random_and_he_initialization ===\n");
    intnn_fc_layer* layer = intnn_fc_create(5, 5);

    // test_random_weight  期望有非零元素
    intnn_fc_set_random_weight(layer);
    int nonzero = 0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (intnn_get_elem(layer->mWeight, i, j) != 0) nonzero++;
        }
    }
    TEST_ASSERT(nonzero > 0, "Random weight produced nonzero entries");

    // test_random_bias 期望有非零元素
    intnn_fc_set_random_bias(layer);
    int nonzero_bias = 0;
    for (int j = 0; j < 5; j++) {
        if (intnn_get_elem(layer->mBias, 0, j) != 0) nonzero_bias++;
    }
    TEST_ASSERT(nonzero_bias > 0, "Random bias produced nonzero entries");

    // test_random_dfa_weight shape
    intnn_fc_set_random_dfa_weight(layer, 4, 3);
    TEST_ASSERT(layer->mDfaWeight != NULL, "DFA weight allocated");
    TEST_ASSERT(intnn_rows(layer->mDfaWeight) == 4 && intnn_cols(layer->mDfaWeight) == 3, 
                "DFA weight shape correct");

    // test_he_init_bias should be zero
    intnn_fc_init_he_weight_bias(layer);
    for (int j = 0; j < 5; j++) {
        TEST_ASSERT(intnn_get_elem(layer->mBias, 0, j) == 0, "He init sets bias to zero");
    }

    intnn_fc_free(layer);
    free(layer);
}

void test_forward_as_is_activation() {
    printf("=== test_forward_as_is_activation ===\n");
    // inDim=2, outDim=2
    intnn_fc_layer* layer = intnn_fc_create(2, 2);


    // weight = [[1,2],[3,4]]
    intnn_mat* W = layer->mWeight;
    intnn_set_elem(W, 0, 0, 1);
    intnn_set_elem(W, 0, 1, 2);
    intnn_set_elem(W, 1, 0, 3);
    intnn_set_elem(W, 1, 1, 4);

    

    // bias = [1,1]
    intnn_mat* B = layer->mBias;
    intnn_set_all_constant(B, 1);

    // 激活函数 = AS_IS
    intnn_fc_set_actv(layer, INTNN_ACTV_AS_IS);
    // 输入 X = [[1,1]]
    intnn_mat* x = intnn_create_mat(1, 2);
    intnn_set_elem(x, 0, 0, 1);
    intnn_set_elem(x, 0, 1, 1);

    intnn_fc_forward(layer, x);
    // 结果 Y = [1*1 + 1*3 +1, 1*2 + 1*4 +1] = [5,7]

    int expected[] = {5, 7};
    check_mat_equal(layer->mOutput, expected, 2, "Forward (AS_IS) Y = [5,7]");

    intnn_free_mat(x);
    intnn_fc_free(layer);
    free(layer);
}

void test_backward_as_is_activation() {
    printf("=== test_backward_as_is_activation ===\n");
    intnn_fc_layer* layer = intnn_fc_create(2, 2);

    // 将 weight, bias 置零
    intnn_mat* W = layer->mWeight;
    intnn_mat* B = layer->mBias;
    intnn_reset_zero(W, 2, 2);
    intnn_reset_zero(B, 1, 2);


    // 激活函数 = AS_IS
    intnn_fc_set_actv(layer, INTNN_ACTV_AS_IS);

    // 输入 X = [[1,1]]
    intnn_mat* x = intnn_create_mat(1, 2);
    intnn_set_elem(x, 0, 0, 1);
    intnn_set_elem(x, 0, 1, 1);
    // 先前向
    intnn_fc_forward(layer, x);
    // lastDeltas = [[1,1]]
    intnn_mat* lastDeltas = intnn_create_mat(1, 2);
    intnn_set_all_constant(lastDeltas, 1);

    intnn_fc_backward(layer, lastDeltas, 1);

    // 期望 weight 更新 = -1 * ([[1],[1]]·[1,1]) = [[-1,-1],[-1,-1]]
    int expectedW[] = {-1, -1, -1, -1};
    check_mat_equal(layer->mWeight, expectedW, 4, "Backward weight updated to [[-1,-1],[-1,-1]]");

    // 期望 bias 更新 = -1 * sum([1,1]) = [-1,-1]
    int expectedB[] = {-1, -1};
    check_mat_equal(layer->mBias, expectedB, 2, "Backward bias updated to [-1,-1]");

    // 检查 mDeltasTranspose shape = (2,1)
    intnn_mat* dT = layer->mDeltasTranspose;
    TEST_ASSERT(intnn_rows(dT) == 2 && intnn_cols(dT) == 1, "DeltasTranspose shape correct (2x1)");

    intnn_free_mat(x);
    intnn_free_mat(lastDeltas);
    intnn_fc_free(layer);
    free(layer);
}

void test_use_batch_and_dfa_flags() {
    printf("=== test_use_batch_and_dfa_flags ===\n");
    intnn_fc_layer* layer = intnn_fc_create(3, 3);

    intnn_fc_use_batch_normalization(layer, true);
    TEST_ASSERT(layer->mUseBn == true, "Enabled batch normalization");
    intnn_fc_use_batch_normalization(layer, false);
    TEST_ASSERT(layer->mUseBn == false, "Disabled batch normalization");

    intnn_fc_use_dfa(layer, true);
    TEST_ASSERT(layer->mUseDfa == true, "Enabled DFA");
    intnn_fc_use_dfa(layer, false);
    TEST_ASSERT(layer->mUseDfa == false, "Disabled DFA");

    intnn_fc_free(layer);
    free(layer);
}

void test_print_functions() {
    printf("=== test_print_functions ===\n");
    intnn_fc_layer* layer = intnn_fc_create(2, 2);

    // 手动设定 weight=[[1,2],[3,4]], bias=[5,6]
    intnn_set_elem(layer->mWeight, 0, 0, 1);
    intnn_set_elem(layer->mWeight, 0, 1, 2);
    intnn_set_elem(layer->mWeight, 1, 0, 3);
    intnn_set_elem(layer->mWeight, 1, 1, 4);
    intnn_set_elem(layer->mBias, 0, 0, 5);
    intnn_set_elem(layer->mBias, 0, 1, 6);

    printf("---- Weight ----\n");
    intnn_fc_print_weight(layer, stdout);
    printf("---- Bias ----\n");
    intnn_fc_print_bias(layer, stdout);

    // 再测试 mInter 与 mOutput
    intnn_mat* x = intnn_create_mat(1, 2);
    intnn_set_elem(x, 0, 0, 1);
    intnn_set_elem(x, 0, 1, 1);
    intnn_fc_set_actv(layer, INTNN_ACTV_AS_IS);
    intnn_fc_forward(layer, x);

    printf("---- Inter ----\n");
    intnn_fc_print_inter(layer, stdout);
    printf("---- Output ----\n");
    intnn_fc_print_output(layer, stdout);

    intnn_free_mat(x);
    intnn_fc_free(layer);
    free(layer);
}

int main() {
    test_create_and_free();
    // test_set_and_get_name();
    test_random_and_he_initialization();
    test_forward_as_is_activation();
    test_backward_as_is_activation();
    test_use_batch_and_dfa_flags();
    test_print_functions();

    printf("All tests passed!\n");
    return 0;
}
