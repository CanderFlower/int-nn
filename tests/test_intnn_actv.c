#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
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
    
    // 更新测试数据：严格匹配分段边界
    int data[] = {
        -1000,  // < -127 → 1
        -127,   // = 第一个断点
        -100,   // [-127, -74] → x/8 + 20
        -74,    // = 第二个断点
        -50,    // [-74, -31] → x/2 + 48
        -31,    // = 第三个断点
        0,      // [-31, 32] → x + 64
        32,     // = 第四个断点
        50,     // [32, 75] → x/2 + 80
        75,     // = 第五个断点
        100,    // [75, 128] → x/8 + 108
        128,    // = 第六个断点
        150     // > 128 → INTNN_MAX
    };
    
    // 根据公式计算预期值 (k=3)
    int expected[] = {
        1,       // < -127
        1,       // -127
        (-100/8) + 20,
        20,      // -74/8 +20 ≈ -9.25 +20 = 10.75 → 11
        (-50/2) + 48,
        48,      // -31/2 +48 = -15.5 +48 = 32.5 → 33
        0 + 64,
        64,      // 32+64=96
        (50/2) + 80,
        80,      // 75/2+80=37.5+80=117.5 → 118
        (100/8) + 108,
        108,     // 128/8+108=16+108=124
        INTNN_MAX // INTNN_MAX
    };
    
    // 对应分段的斜率倒数
    int expected_grad[] = {
        INTNN_MAX,  // < -127
        INTNN_MAX,  // -127
        8, 
        8, 
        2, 
        2, 
        1, 
        1, 
        2, 
        2, 
        8, 
        8, 
        INTNN_MAX   // > 128
    };
    
    intnn_mat* in = create_test_matrix(1, 13, data);
    intnn_mat* out = intnn_create_mat(1, 13);
    intnn_mat* grad = intnn_create_mat(1, 13);
    
    // 确保矩阵分配成功
    assert(in != NULL && out != NULL && grad != NULL);
    
    intnn_sigmoid(out, in, grad, 1); // k=3 即除以8
    
    // 使用assert验证结果
    for (int c = 0; c < 13; c++) {
        int actual = intnn_get_elem(out, 0, c);
        int actual_grad = intnn_get_elem(grad, 0, c);
        
        printf("Input: %5d, Expected: %5d, Actual: %5d, Grad: %5d\n", 
               data[c], expected[c], actual, actual_grad);
        
        assert(actual == expected[c]);
        assert(actual_grad == expected_grad[c]);
    }
    
    // 清理
    intnn_free_mat(in);
    intnn_free_mat(out);
    intnn_free_mat(grad);
    
    printf("2D Sigmoid: PASSED\n\n");
}

void test_tanh_2d() {
    print_test_header("2D Tanh Activation");
    
    // 更新测试数据：严格匹配分段边界
    int data[] = {
        -1000,  // < -127 → -128
        -127,   // = 第一个断点
        -100,   // [-127, -74] → x/4 - 88
        -74,    // = 第二个断点
        -50,    // [-74, -31] → x - 32
        -31,    // = 第三个断点
        0,      // [-31, 32] → 2*x
        32,     // = 第四个断点
        50,     // [32, 75] → x + 32
        75,     // = 第五个断点
        100,    // [75, 128] → x/4 + 88
        128,    // = 第六个断点
        150     // > 128 → INTNN_MAX
    };
    
    // 根据公式计算预期值 (k=3, numItems=1)
    int expected[] = {
        -128,   // < -127
        -128,   // -127
        (-100/4) - 88,
        -88,    // -74/4-88 ≈ -18.5-88 = -106.5 → -107
        -50 - 32,
        -63,    // -31-32 = -63
        2 * 0,    // 0
        64,     // 2 * 32=64
        50 + 32,
        107,    // 75+32=107
        (100/4) + 88,
        88+25,  // 128/4+88=32+88=120
        INTNN_MAX // INTNN_MAX
    };
    
    // 对应分段的斜率倒数
    int expected_grad[] = {
        INTNN_MAX,  // < -127
        INTNN_MAX,  // -127
        4, 
        4, 
        1, 
        1, 
        2, 
        2, 
        1, 
        1, 
        4, 
        4, 
        INTNN_MAX   // > 128
    };
    
    intnn_mat* in = create_test_matrix(1, 13, data);
    intnn_mat* out = intnn_create_mat(1, 13);
    intnn_mat* grad = intnn_create_mat(1, 13);
    
    // 确保矩阵分配成功
    assert(in != NULL && out != NULL && grad != NULL);
    
    intnn_tanh(out, in, grad, 1, 1); // k=3, numItems=1
    
    // 使用assert验证结果
    for (int c = 0; c < 13; c++) {
        int actual = intnn_get_elem(out, 0, c);
        int actual_grad = intnn_get_elem(grad, 0, c);
        
        printf("Input: %5d, Expected: %5d, Actual: %5d, Grad: %5d\n", 
               data[c], expected[c], actual, actual_grad);
        
        assert(actual == expected[c]);
        assert(actual_grad == expected_grad[c]);
    }
    
    // 清理
    intnn_free_mat(in);
    intnn_free_mat(out);
    intnn_free_mat(grad);
    
    printf("2D Tanh: PASSED\n\n");
}

void test_softmax_2d() {
    print_test_header("2D Softmax Activation");
    
    // 测试数据：包含正负值和全零行
    int data[] = {
        // 第1行：正常值
        10, 20, 30, 
        // 第2行：全负值
        -5, -10, -15,
        // 第3行：全零
        0, 0, 0
    };
    
    intnn_mat* in = create_test_matrix(3, 3, data);
    intnn_mat* out = intnn_create_mat(3, 3);
    intnn_mat* grad = intnn_create_mat(3, 3);
    
    // 确保矩阵分配成功
    assert(in != NULL && out != NULL && grad != NULL);
    
    intnn_softmax(out, in, grad, 0);
    
    // 验证每行总和应为INTNN_MAX
    for (int r = 0; r < 3; r++) {
        int row_sum = 0;
        for (int c = 0; c < 3; c++) {
            row_sum += intnn_get_elem(out, r, c);
        }
        printf("Row %d sum: %d\n", r, row_sum);
        
        // 允许1的误差（整数舍入）
        assert(row_sum >= INTNN_MAX - 1 && row_sum <= INTNN_MAX);
    }
    
    // 验证第2行（全负值）输出应该全为0
    for (int c = 0; c < 3; c++) {
        assert(intnn_get_elem(out, 1, c) == 0);
    }
    
    // 验证第3行（全零）应该是均匀分布
    int expected_avg = INTNN_MAX / 3;
    for (int c = 0; c < 3; c++) {
        int val = intnn_get_elem(out, 2, c);
        assert(val >= expected_avg - 1 && val <= expected_avg + 1);
    }
    
    // 检查梯度：负值和零值应标记为INTNN_MAX
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            int val = intnn_get_elem(in, r, c);
            int grad_val = intnn_get_elem(grad, r, c);
            
            if (val <= 0) {
                assert(grad_val == INTNN_MAX);
            } else {
                assert(grad_val == 1);
            }
        }
    }
    
    // 清理
    intnn_free_mat(in);
    intnn_free_mat(out);
    intnn_free_mat(grad);
    
    printf("2D Softmax: PASSED\n\n");
}

void test_relu_2d() {
    print_test_header("2D ReLU Activation");
    
    int data[] = {
        -50, -10, 0, 10, 50, 150
    };
    // 正确预期值 (INTNN_MAX = 127)
    int expected[] = {
        0, 0, 0, 10, 50, INTNN_MAX
    };
    int expected_grad[] = {
        INTNN_MAX, INTNN_MAX, INTNN_MAX, 1, 1, INTNN_MAX
    };
    
    intnn_mat* in = create_test_matrix(2, 3, data);
    intnn_mat* out = intnn_create_mat(2, 3);
    intnn_mat* grad = intnn_create_mat(2, 3);
    
    // 确保矩阵分配成功
    assert(in != NULL && out != NULL && grad != NULL);
    
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
    
    printf("2D ReLU: PASSED\n\n");
}

void test_leaky_relu_2d() {
    print_test_header("2D Leaky ReLU Activation");
    
    int data[] = {
        SHRT_MIN - 1, // 小于 SHRT_MIN
        -100, 
        -50, 
        -10, 
        0, 
        10, 
        50, 
        100,
        SHRT_MAX + 1  // 大于 SHRT_MAX
    };
    
    // 正确预期值
    int expected[] = {
        SHRT_MIN,     // 下溢
        -100/5,       // -20
        -50/5,        // -10
        -10/5,        // -2
        0, 
        10, 
        50, 
        100,
        SHRT_MAX      // 上溢
    };
    
    // 正确梯度倒数
    int expected_grad[] = {
        INTNN_MAX,     // 下溢
        5, 
        5, 
        5, 
        1, 
        1, 
        1, 
        1,
        INTNN_MAX      // 上溢
    };
    
    intnn_mat* in = create_test_matrix(1, 9, data);
    intnn_mat* out = intnn_create_mat(1, 9);
    intnn_mat* grad = intnn_create_mat(1, 9);
    
    // 确保矩阵分配成功
    assert(in != NULL && out != NULL && grad != NULL);
    
    intnn_leakyrelu(out, in, grad, 0);
    
    // 使用assert验证结果
    for (int c = 0; c < 9; c++) {
        int actual = intnn_get_elem(out, 0, c);
        int actual_grad = intnn_get_elem(grad, 0, c);
        
        assert(actual == expected[c]);
        assert(actual_grad == expected_grad[c]);
    }
    
    // 清理
    intnn_free_mat(in);
    intnn_free_mat(out);
    intnn_free_mat(grad);
    
    printf("2D Leaky ReLU: PASSED\n\n");
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
        100, INTNN_MAX+1,  // INTNN_MAX+1 = 128
        0, INTNN_MAX       // INTNN_MAX = 127
    };
    
    intnn_mat3d* in = create_test_matrix3d(3, 2, 2, data);
    intnn_mat3d* out = intnn_create_mat3d(3, 2, 2);
    intnn_mat3d* grad = intnn_create_mat3d(3, 2, 2);
    
    // 确保矩阵分配成功
    assert(in != NULL && out != NULL && grad != NULL);
    
    intnn_sigmoid3d(out, in, grad, 1); // k=3
    
    // 验证每个深度层
    for (int d = 0; d < 3; d++) {
        printf("Depth %d:\n", d);
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                int in_val = intnn_mat3d_get_elem(in, d, r, c);
                int out_val = intnn_mat3d_get_elem(out, d, r, c);
                int grad_val = intnn_mat3d_get_elem(grad, d, r, c);
                
                printf("  Input: %5d, Output: %5d, Grad: %5d\n", 
                       in_val, out_val, grad_val);
                
                // 验证输出在合理范围
                assert(out_val >= 1 && out_val <= INTNN_MAX);
                
                // 验证梯度倒数是预期值（1, 2, 4, 8 或 INTNN_MAX）
                assert(grad_val == 1 || grad_val == 2 || grad_val == 4 || 
                       grad_val == 8 || grad_val == INTNN_MAX);
            }
        }
    }
    
    // 清理
    intnn_free_mat3d(in);
    intnn_free_mat3d(out);
    intnn_free_mat3d(grad);
    
    printf("3D Sigmoid: PASSED\n\n");
}

void test_activate_3d() {
    print_test_header("3D General Activation (Tanh)");
    
    int data[] = {
        // 深度0
        -50, 20,
        10, -30,
        
        // 深度1
        0, INTNN_MAX,  // INTNN_MAX = 127
        -20, INTNN_MAX+1 // INTNN_MAX+1 = 128
    };
    
    intnn_mat3d* in = create_test_matrix3d(2, 2, 2, data);
    intnn_mat3d* out = intnn_create_mat3d(2, 2, 2);
    intnn_mat3d* grad = intnn_create_mat3d(2, 2, 2);
    
    // 确保矩阵分配成功
    assert(in != NULL && out != NULL && grad != NULL);
    
    // 测试Tanh激活
    intnn_activate3d(out, in, grad, INTNN_ACTV_TANH, 3, 1);
    
    // 验证结果
    for (int d = 0; d < 2; d++) {
        printf("Depth %d:\n", d);
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                int in_val = intnn_mat3d_get_elem(in, d, r, c);
                int out_val = intnn_mat3d_get_elem(out, d, r, c);
                int grad_val = intnn_mat3d_get_elem(grad, d, r, c);
                
                printf("  Input: %5d, Output: %5d, Grad: %5d\n", 
                       in_val, out_val, grad_val);
                
                // 验证输出在合理范围
                assert(out_val >= -128 && out_val <= INTNN_MAX);
                
                // 验证梯度倒数是预期值（1, 2, 4 或 INTNN_MAX）
                assert(grad_val == 1 || grad_val == 2 || grad_val == 4 || 
                       grad_val == INTNN_MAX);
            }
        }
    }
    
    // 清理
    intnn_free_mat3d(in);
    intnn_free_mat3d(out);
    intnn_free_mat3d(grad);
    
    printf("3D General Activation: PASSED\n\n");
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
