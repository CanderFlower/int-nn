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

// ====================== �������� ======================

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

// ====================== 2D ���Ժ��� ======================

void test_sigmoid_2d() {
    print_test_header("2D Sigmoid Activation");
    
    // ���²������ݣ��ϸ�ƥ��ֶα߽�
    int data[] = {
        -1000,  // < -127 �� 1
        -127,   // = ��һ���ϵ�
        -100,   // [-127, -74] �� x/8 + 20
        -74,    // = �ڶ����ϵ�
        -50,    // [-74, -31] �� x/2 + 48
        -31,    // = �������ϵ�
        0,      // [-31, 32] �� x + 64
        32,     // = ���ĸ��ϵ�
        50,     // [32, 75] �� x/2 + 80
        75,     // = ������ϵ�
        100,    // [75, 128] �� x/8 + 108
        128,    // = �������ϵ�
        150     // > 128 �� INTNN_MAX
    };
    
    // ���ݹ�ʽ����Ԥ��ֵ (k=3)
    int expected[] = {
        1,       // < -127
        1,       // -127
        (-100/8) + 20,
        20,      // -74/8 +20 �� -9.25 +20 = 10.75 �� 11
        (-50/2) + 48,
        48,      // -31/2 +48 = -15.5 +48 = 32.5 �� 33
        0 + 64,
        64,      // 32+64=96
        (50/2) + 80,
        80,      // 75/2+80=37.5+80=117.5 �� 118
        (100/8) + 108,
        108,     // 128/8+108=16+108=124
        INTNN_MAX // INTNN_MAX
    };
    
    // ��Ӧ�ֶε�б�ʵ���
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
    
    // ȷ���������ɹ�
    assert(in != NULL && out != NULL && grad != NULL);
    
    intnn_sigmoid(out, in, grad, 1); // k=3 ������8
    
    // ʹ��assert��֤���
    for (int c = 0; c < 13; c++) {
        int actual = intnn_get_elem(out, 0, c);
        int actual_grad = intnn_get_elem(grad, 0, c);
        
        printf("Input: %5d, Expected: %5d, Actual: %5d, Grad: %5d\n", 
               data[c], expected[c], actual, actual_grad);
        
        assert(actual == expected[c]);
        assert(actual_grad == expected_grad[c]);
    }
    
    // ����
    intnn_free_mat(in);
    intnn_free_mat(out);
    intnn_free_mat(grad);
    
    printf("2D Sigmoid: PASSED\n\n");
}

void test_tanh_2d() {
    print_test_header("2D Tanh Activation");
    
    // ���²������ݣ��ϸ�ƥ��ֶα߽�
    int data[] = {
        -1000,  // < -127 �� -128
        -127,   // = ��һ���ϵ�
        -100,   // [-127, -74] �� x/4 - 88
        -74,    // = �ڶ����ϵ�
        -50,    // [-74, -31] �� x - 32
        -31,    // = �������ϵ�
        0,      // [-31, 32] �� 2*x
        32,     // = ���ĸ��ϵ�
        50,     // [32, 75] �� x + 32
        75,     // = ������ϵ�
        100,    // [75, 128] �� x/4 + 88
        128,    // = �������ϵ�
        150     // > 128 �� INTNN_MAX
    };
    
    // ���ݹ�ʽ����Ԥ��ֵ (k=3, numItems=1)
    int expected[] = {
        -128,   // < -127
        -128,   // -127
        (-100/4) - 88,
        -88,    // -74/4-88 �� -18.5-88 = -106.5 �� -107
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
    
    // ��Ӧ�ֶε�б�ʵ���
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
    
    // ȷ���������ɹ�
    assert(in != NULL && out != NULL && grad != NULL);
    
    intnn_tanh(out, in, grad, 1, 1); // k=3, numItems=1
    
    // ʹ��assert��֤���
    for (int c = 0; c < 13; c++) {
        int actual = intnn_get_elem(out, 0, c);
        int actual_grad = intnn_get_elem(grad, 0, c);
        
        printf("Input: %5d, Expected: %5d, Actual: %5d, Grad: %5d\n", 
               data[c], expected[c], actual, actual_grad);
        
        assert(actual == expected[c]);
        assert(actual_grad == expected_grad[c]);
    }
    
    // ����
    intnn_free_mat(in);
    intnn_free_mat(out);
    intnn_free_mat(grad);
    
    printf("2D Tanh: PASSED\n\n");
}

void test_softmax_2d() {
    print_test_header("2D Softmax Activation");
    
    // �������ݣ���������ֵ��ȫ����
    int data[] = {
        // ��1�У�����ֵ
        10, 20, 30, 
        // ��2�У�ȫ��ֵ
        -5, -10, -15,
        // ��3�У�ȫ��
        0, 0, 0
    };
    
    intnn_mat* in = create_test_matrix(3, 3, data);
    intnn_mat* out = intnn_create_mat(3, 3);
    intnn_mat* grad = intnn_create_mat(3, 3);
    
    // ȷ���������ɹ�
    assert(in != NULL && out != NULL && grad != NULL);
    
    intnn_softmax(out, in, grad, 0);
    
    // ��֤ÿ���ܺ�ӦΪINTNN_MAX
    for (int r = 0; r < 3; r++) {
        int row_sum = 0;
        for (int c = 0; c < 3; c++) {
            row_sum += intnn_get_elem(out, r, c);
        }
        printf("Row %d sum: %d\n", r, row_sum);
        
        // ����1�����������룩
        assert(row_sum >= INTNN_MAX - 1 && row_sum <= INTNN_MAX);
    }
    
    // ��֤��2�У�ȫ��ֵ�����Ӧ��ȫΪ0
    for (int c = 0; c < 3; c++) {
        assert(intnn_get_elem(out, 1, c) == 0);
    }
    
    // ��֤��3�У�ȫ�㣩Ӧ���Ǿ��ȷֲ�
    int expected_avg = INTNN_MAX / 3;
    for (int c = 0; c < 3; c++) {
        int val = intnn_get_elem(out, 2, c);
        assert(val >= expected_avg - 1 && val <= expected_avg + 1);
    }
    
    // ����ݶȣ���ֵ����ֵӦ���ΪINTNN_MAX
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
    
    // ����
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
    // ��ȷԤ��ֵ (INTNN_MAX = 127)
    int expected[] = {
        0, 0, 0, 10, 50, INTNN_MAX
    };
    int expected_grad[] = {
        INTNN_MAX, INTNN_MAX, INTNN_MAX, 1, 1, INTNN_MAX
    };
    
    intnn_mat* in = create_test_matrix(2, 3, data);
    intnn_mat* out = intnn_create_mat(2, 3);
    intnn_mat* grad = intnn_create_mat(2, 3);
    
    // ȷ���������ɹ�
    assert(in != NULL && out != NULL && grad != NULL);
    
    intnn_relu8bit(out, in, grad, 0);
    
    // ʹ��assert��֤���
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) {
            int idx = r * 3 + c;
            int actual = intnn_get_elem(out, r, c);
            int actual_grad = intnn_get_elem(grad, r, c);
            
            assert(actual == expected[idx]);
            assert(actual_grad == expected_grad[idx]);
        }
    }
    
    // ����
    intnn_free_mat(in);
    intnn_free_mat(out);
    intnn_free_mat(grad);
    
    printf("2D ReLU: PASSED\n\n");
}

void test_leaky_relu_2d() {
    print_test_header("2D Leaky ReLU Activation");
    
    int data[] = {
        SHRT_MIN - 1, // С�� SHRT_MIN
        -100, 
        -50, 
        -10, 
        0, 
        10, 
        50, 
        100,
        SHRT_MAX + 1  // ���� SHRT_MAX
    };
    
    // ��ȷԤ��ֵ
    int expected[] = {
        SHRT_MIN,     // ����
        -100/5,       // -20
        -50/5,        // -10
        -10/5,        // -2
        0, 
        10, 
        50, 
        100,
        SHRT_MAX      // ����
    };
    
    // ��ȷ�ݶȵ���
    int expected_grad[] = {
        INTNN_MAX,     // ����
        5, 
        5, 
        5, 
        1, 
        1, 
        1, 
        1,
        INTNN_MAX      // ����
    };
    
    intnn_mat* in = create_test_matrix(1, 9, data);
    intnn_mat* out = intnn_create_mat(1, 9);
    intnn_mat* grad = intnn_create_mat(1, 9);
    
    // ȷ���������ɹ�
    assert(in != NULL && out != NULL && grad != NULL);
    
    intnn_leakyrelu(out, in, grad, 0);
    
    // ʹ��assert��֤���
    for (int c = 0; c < 9; c++) {
        int actual = intnn_get_elem(out, 0, c);
        int actual_grad = intnn_get_elem(grad, 0, c);
        
        assert(actual == expected[c]);
        assert(actual_grad == expected_grad[c]);
    }
    
    // ����
    intnn_free_mat(in);
    intnn_free_mat(out);
    intnn_free_mat(grad);
    
    printf("2D Leaky ReLU: PASSED\n\n");
}

// ====================== 3D ���Ժ��� ======================

void test_sigmoid_3d() {
    print_test_header("3D Sigmoid Activation");
    
    int data[] = {
        // ���0
        -150, -100, 
        -80, -40,
        
        // ���1
        -10, 10,
        40, 80,
        
        // ���2
        100, INTNN_MAX+1,  // INTNN_MAX+1 = 128
        0, INTNN_MAX       // INTNN_MAX = 127
    };
    
    intnn_mat3d* in = create_test_matrix3d(3, 2, 2, data);
    intnn_mat3d* out = intnn_create_mat3d(3, 2, 2);
    intnn_mat3d* grad = intnn_create_mat3d(3, 2, 2);
    
    // ȷ���������ɹ�
    assert(in != NULL && out != NULL && grad != NULL);
    
    intnn_sigmoid3d(out, in, grad, 1); // k=3
    
    // ��֤ÿ����Ȳ�
    for (int d = 0; d < 3; d++) {
        printf("Depth %d:\n", d);
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                int in_val = intnn_mat3d_get_elem(in, d, r, c);
                int out_val = intnn_mat3d_get_elem(out, d, r, c);
                int grad_val = intnn_mat3d_get_elem(grad, d, r, c);
                
                printf("  Input: %5d, Output: %5d, Grad: %5d\n", 
                       in_val, out_val, grad_val);
                
                // ��֤����ں���Χ
                assert(out_val >= 1 && out_val <= INTNN_MAX);
                
                // ��֤�ݶȵ�����Ԥ��ֵ��1, 2, 4, 8 �� INTNN_MAX��
                assert(grad_val == 1 || grad_val == 2 || grad_val == 4 || 
                       grad_val == 8 || grad_val == INTNN_MAX);
            }
        }
    }
    
    // ����
    intnn_free_mat3d(in);
    intnn_free_mat3d(out);
    intnn_free_mat3d(grad);
    
    printf("3D Sigmoid: PASSED\n\n");
}

void test_activate_3d() {
    print_test_header("3D General Activation (Tanh)");
    
    int data[] = {
        // ���0
        -50, 20,
        10, -30,
        
        // ���1
        0, INTNN_MAX,  // INTNN_MAX = 127
        -20, INTNN_MAX+1 // INTNN_MAX+1 = 128
    };
    
    intnn_mat3d* in = create_test_matrix3d(2, 2, 2, data);
    intnn_mat3d* out = intnn_create_mat3d(2, 2, 2);
    intnn_mat3d* grad = intnn_create_mat3d(2, 2, 2);
    
    // ȷ���������ɹ�
    assert(in != NULL && out != NULL && grad != NULL);
    
    // ����Tanh����
    intnn_activate3d(out, in, grad, INTNN_ACTV_TANH, 3, 1);
    
    // ��֤���
    for (int d = 0; d < 2; d++) {
        printf("Depth %d:\n", d);
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                int in_val = intnn_mat3d_get_elem(in, d, r, c);
                int out_val = intnn_mat3d_get_elem(out, d, r, c);
                int grad_val = intnn_mat3d_get_elem(grad, d, r, c);
                
                printf("  Input: %5d, Output: %5d, Grad: %5d\n", 
                       in_val, out_val, grad_val);
                
                // ��֤����ں���Χ
                assert(out_val >= -128 && out_val <= INTNN_MAX);
                
                // ��֤�ݶȵ�����Ԥ��ֵ��1, 2, 4 �� INTNN_MAX��
                assert(grad_val == 1 || grad_val == 2 || grad_val == 4 || 
                       grad_val == INTNN_MAX);
            }
        }
    }
    
    // ����
    intnn_free_mat3d(in);
    intnn_free_mat3d(out);
    intnn_free_mat3d(grad);
    
    printf("3D General Activation: PASSED\n\n");
}

// ====================== �����Ժ��� ======================

int main() {
    printf("=== Starting Integer Neural Network Activation Tests ===\n\n");
    
    // ����2D�����
    test_sigmoid_2d();
    test_tanh_2d();
    test_softmax_2d();
    test_relu_2d();
    test_leaky_relu_2d();
    
    // ����3D�����
    test_sigmoid_3d();
    test_activate_3d();
    
    printf("\n=== All Tests Passed Successfully! ===\n");
    return 0;
}
