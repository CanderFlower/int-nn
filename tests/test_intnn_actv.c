#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "intnn_mat.h"
#include "intnn_mat3d.h"
#include "intnn_actv.h"
#include "intnn_consts.h"

#ifndef INTNN_MAX
#define INTNN_MAX 127
#endif
// ====================== 辅助函数 ======================

void print_test_header(const char* test_name) {
    printf("=======================================\n");
    printf("TEST: %s\n", test_name);
    printf("=======================================\n");
}

intnn_mat* create_test_matrix(int rows, int cols, int* data) {
    intnn_mat* mat = intnn_create_mat(rows, cols);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            intnn_set_elem(mat, r, c, data[idx]);
        }
    }
    return mat;
}

intnn_mat3d* create_test_matrix3d(int depth, int rows, int cols, int* data) {
    intnn_mat3d* mat3d = intnn_create_mat3d(depth, rows, cols);
    for (int d = 0; d < depth; d++) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int idx = d * (rows * cols) + r * cols + c;
                intnn_mat3d_set_elem(mat3d, d, r, c, data[idx]);
            }
        }
    }
    return mat3d;
}

// ====================== 2D 测试函数 ======================

void test_sigmoid_2d() {
    print_test_header("2D Sigmoid Activation");
    
    // 测试数据：覆盖所有分段
    int data[] = {
        -150, -100, -80, -40, -10, 10, 40, 80, 100, 150
    };
    int expected[] = {
        1, 1, 20, 48, 64, 64, 80, 108, 127, 127
    };
    int expected_grad[] = {
        INTNN_MAX, 8, 2, 1, 1, 1, 2, 8, INTNN_MAX, INTNN_MAX
    };
    
    intnn_mat* in = create_test_matrix(2, 5, data);
    intnn_mat* out = intnn_create_mat(2, 5);
    intnn_mat* grad = intnn_create_mat(2, 5);
    
    intnn_sigmoid(out, in, grad, 3); // k=3 即除以8
    
    // 使用assert验证结果
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 5; c++) {
            int idx = r * 5 + c;
            int actual = intnn_get_elem(out, r, c);
            int actual_grad = intnn_get_elem(grad, r, c);
            
            assert(actual == expected[idx]);
            assert(actual_grad == expected_grad[idx]);
        }
    }
    
    // 清理
    intnn_free_mat(in);
    intnn_free_mat(out);
    intnn_free_mat(grad);
    
    printf("2D Sigmoid: PASSED\n");
}

void test_tanh_2d() {
    print_test_header("2D Tanh Activation");
    
    // 测试数据：覆盖所有分段
    int data[] = {
        -150, -100, -80, -40, -10, 10, 40, 80, 100, 150
    };
    int expected[] = {
        -128, -128, -108, -72, -20, 20, 72, 108, 127, 127
    };
    int expected_grad[] = {
        INTNN_MAX, 4, 1, 1, 1, 1, 1, 4, INTNN_MAX, INTNN_MAX
    };
    
    intnn_mat* in = create_test_matrix(2, 5, data);
    intnn_mat* out = intnn_create_mat(2, 5);
    intnn_mat* grad = intnn_create_mat(2, 5);
    
    intnn_tanh(out, in, grad, 3, 1); // k=3, numItems=1
    
    // 使用assert验证结果
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 5; c++) {
            int idx = r * 5 + c;
            int actual = intnn_get_elem(out, r, c);
            int actual_grad = intnn_get_elem(grad, r, c);
            
            assert(actual == expected[idx]);
            assert(actual_grad == expected_grad[idx]);
        }
    }
    
    // 清理
    intnn_free_mat(in);
    intnn_free_mat(out);
    intnn_free_mat(grad);
    
    printf("2D Tanh: PASSED\n");
}

void test_softmax_2d() {
    print_test_header("2D Softmax Activation");
    
    // 测试数据：包含正负值
    int data[] = {
        10, 20, 30, 
        -5, 15, 25
    };
    
    intnn_mat* in = create_test_matrix(2, 3, data);
    intnn_mat* out = intnn_create_mat(2, 3);
    intnn_mat* grad = intnn_create_mat(2, 3);
    
    intnn_softmax(out, in, grad, 0);
    
    // 验证行1：总和应为INTNN_MAX
    int row0_sum = intnn_sum(out);
    assert(row0_sum >= INTNN_MAX - 10 && row0_sum <= INTNN_MAX);
    
    int row1_sum = 0;
    for (int c = 0; c < 3; c++) {
        row1_sum += intnn_get_elem(out, 1, c);
    }
    assert(row1_sum >= INTNN_MAX - 10 && row1_sum <= INTNN_MAX);
    
    // 检查梯度
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) {
            int val = intnn_get_elem(in, r, c);
            int grad_val = intnn_get_elem(grad, r, c);
            
            if (val <= 0) {
                assert(grad_val == INTNN_MAX);
            }
        }
    }
    
    // 清理
    intnn_free_mat(in);
    intnn_free_mat(out);
    intnn_free_mat(grad);
    
    printf("2D Softmax: PASSED\n");
}

void test_relu_2d() {
    print_test_header("2D ReLU Activation");
    
    int data[] = {
        -50, -10, 0, 10, 50, 150
    };
    int expected[] = {
        0, 0, 0, 10, 50, 127
    };
    int expected_grad[] = {
        INTNN_MAX, INTNN_MAX, INTNN_MAX, 1, 1, INTNN_MAX
    };
    
    intnn_mat* in = create_test_matrix(2, 3, data);
    intnn_mat* out = intnn_create_mat(2, 3);
    intnn_mat* grad = intnn_create_mat(2, 3);
    
    intnn_relu8bit(out, in, grad, 0);
    
    // 使用assert验证结果
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) {
            int idx = r * 3 + c;
            int actual = intnn_get_elem(out, r, c);
            int actual_grad = intnn_get_elem(grad, r, c);
            
            assert(actual == expected[idx]);
            assert(actual_grad == expected_grad[idx]);
        }
    }
    
    // 清理
    intnn_free_mat(in);
    intnn_free_mat(out);
    intnn_free_mat(grad);
    
    printf("2D ReLU: PASSED\n");
}

void test_leaky_relu_2d() {
    print_test_header("2D Leaky ReLU Activation");
    
    int data[] = {
        -100, -50, -10, 0, 10, 50, 100
    };
    int expected[] = {
        -32768, -10, -2, 0, 10, 50, 32767
    };
    int expected_grad[] = {
        INTNN_MAX, 5, 5, 1, 1, 1, INTNN_MAX
    };
    
    intnn_mat* in = create_test_matrix(1, 7, data);
    intnn_mat* out = intnn_create_mat(1, 7);
    intnn_mat* grad = intnn_create_mat(1, 7);
    
    intnn_leakyrelu(out, in, grad, 0);
    
    // 使用assert验证结果
    for (int c = 0; c < 7; c++) {
        int actual = intnn_get_elem(out, 0, c);
        int actual_grad = intnn_get_elem(grad, 0, c);
        
        assert(actual == expected[c]);
        assert(actual_grad == expected_grad[c]);
    }
    
    // 清理
    intnn_free_mat(in);
    intnn_free_mat(out);
    intnn_free_mat(grad);
    
    printf("2D Leaky ReLU: PASSED\n");
}

// ====================== 3D 测试函数 ======================

void test_sigmoid_3d() {
    print_test_header("3D Sigmoid Activation");
    
    int data[] = {
        // 深度0
        -150, -100, 
        -80, -40,
        
        // 深度1
        -10, 10,
        40, 80,
        
        // 深度2
        100, 150,
        0, 0
    };
    
    intnn_mat3d* in = create_test_matrix3d(3, 2, 2, data);
    intnn_mat3d* out = intnn_create_mat3d(3, 2, 2);
    intnn_mat3d* grad = intnn_create_mat3d(3, 2, 2);
    
    intnn_sigmoid3d(out, in, grad, 3); // k=3
    
    // 验证每个深度层
    for (int d = 0; d < 3; d++) {
        intnn_mat* layer_out = intnn_mat3d_get_mat_at_depth(out, d);
        intnn_mat* layer_grad = intnn_mat3d_get_mat_at_depth(grad, d);
        
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                int val = intnn_get_elem(layer_out, r, c);
                int grad_val = intnn_get_elem(layer_grad, r, c);
                
                assert(val >= 1 && val <= INTNN_MAX);
                assert(grad_val > 0 && grad_val < INTNN_MAX);
            }
        }
    }
    
    // 清理
    intnn_free_mat3d(in);
    intnn_free_mat3d(out);
    intnn_free_mat3d(grad);
    
    printf("3D Sigmoid: PASSED\n");
}

void test_activate_3d() {
    print_test_header("3D General Activation");
    
    int data[] = {
        // 深度0
        -50, 20,
        10, -30,
        
        // 深度1
        0, 100,
        -20, 50
    };
    
    intnn_mat3d* in = create_test_matrix3d(2, 2, 2, data);
    intnn_mat3d* out = intnn_create_mat3d(2, 2, 2);
    intnn_mat3d* grad = intnn_create_mat3d(2, 2, 2);
    
    // 测试Tanh激活
    intnn_activate3d(out, in, grad, INTNN_ACTV_TANH, 3, 1);
    
    // 验证结果
    for (int d = 0; d < 2; d++) {
        intnn_mat* layer_out = intnn_mat3d_get_mat_at_depth(out, d);
        intnn_mat* layer_grad = intnn_mat3d_get_mat_at_depth(grad, d);
        
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                int val = intnn_get_elem(layer_out, r, c);
                int grad_val = intnn_get_elem(layer_grad, r, c);
                
                assert(val >= INTNN_MIN && val <= INTNN_MAX);
                assert(grad_val > 0 && grad_val < INTNN_MAX);
            }
        }
    }
    
    // 清理
    intnn_free_mat3d(in);
    intnn_free_mat3d(out);
    intnn_free_mat3d(grad);
    
    printf("3D General Activation: PASSED\n");
}

// ====================== 主测试函数 ======================

int main() {
    printf("=== Starting Integer Neural Network Activation Tests ===\n\n");
    
    // 测试2D激活函数
    test_sigmoid_2d();
    test_tanh_2d();
    test_softmax_2d();
    test_relu_2d();
    test_leaky_relu_2d();
    
    // 测试3D激活函数
    test_sigmoid_3d();
    test_activate_3d();
    
    printf("\n=== All Tests Passed Successfully! ===\n");
    return 0;
}
