#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <limits.h>
#include "intnn_actv.h"
#include "intnn_mat.h"
#include "intnn_mat3d.h"
#include "intnn_consts.h"

// Define constants if not in header
#ifndef INTNN_MIN
#define INTNN_MIN -128
#endif
#ifndef INTNN_MAX
#define INTNN_MAX 127
#endif
#ifndef SHRT_MIN
#define SHRT_MIN (-32768)
#endif
#ifndef SHRT_MAX
#define SHRT_MAX 32767
#endif

// Helper function for clamping values
static int clamp(int input, int minVal, int maxVal) {
    if (input < minVal) return minVal;
    if (input > maxVal) return maxVal;
    return input;
}

// 2D Activation Functions -----------------------------------------------------

void intnn_activate(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv,
                    intnn_actv_type actv, int k, int numItems) {
    // Reset output if dimensions mismatch
    

    if (!intnn_dims_equal(matOut, matIn)) {

        intnn_reset_zero(matOut, intnn_rows(matIn), intnn_cols(matIn));
    }
    
    switch (actv) {
        case INTNN_ACTV_SIGMOID:
            intnn_sigmoid(matOut, matIn, matActvGradInv, k);
            break;
        case INTNN_ACTV_TANH:
            intnn_tanh(matOut, matIn, matActvGradInv, k, numItems);
            break;
        case INTNN_ACTV_RESCALE:
            intnn_rescale(matOut, matIn, matActvGradInv, k);
            break;
        case INTNN_ACTV_SOFTMAX:
            intnn_softmax(matOut, matIn, matActvGradInv, k);
            break;
        case INTNN_ACTV_RELU8BIT:
            intnn_relu8bit(matOut, matIn, matActvGradInv, k);
            break;
        case INTNN_ACTV_LEAKYRELU:
            intnn_leakyrelu(matOut, matIn, matActvGradInv, k);
            break;
        case INTNN_ACTV_PLU:
            intnn_plu(matOut, matIn, matActvGradInv, k);
            break;
        case INTNN_ACTV_AS_IS:
            intnn_as_is(matOut, matIn, matActvGradInv, k);
            break;
        default:
            printf("Unsupported activation type\n");
            break;
    }
}

void intnn_sigmoid(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k) {
    const int yMin = 1;
    const int yMax = INTNN_MAX;
    const int joints[] = {-127, -74, -31, 32, 75, 128};
    const int slopesInv[] = {INTNN_MAX, 8, 2, 1, 2, 8, INTNN_MAX};
    const int divisor = 1 << k;

    for (int r = 0; r < intnn_rows(matOut); r++) {
        for (int c = 0; c < intnn_cols(matOut); c++) {
            int x = intnn_get_elem(matIn, r, c) / divisor;
            int y, grad;
            
            if (x < joints[0])       { y = yMin; grad = slopesInv[0]; }
            else if (x < joints[1])  { y = x/8 + 20; grad = slopesInv[1]; }
            else if (x < joints[2])  { y = x/2 + 48; grad = slopesInv[2]; }
            else if (x < joints[3])  { y = x + 64; grad = slopesInv[3]; }
            else if (x < joints[4])  { y = x/2 + 80; grad = slopesInv[4]; }
            else if (x < joints[5])  { y = x/8 + 108; grad = slopesInv[5]; }
            else                     { y = yMax; grad = slopesInv[6]; }
            
            intnn_set_elem(matOut, r, c, clamp(y, yMin, yMax));
            intnn_set_elem(matActvGradInv, r, c, grad);
        }
    }
}

void intnn_tanh(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k, int numItems) {
    const int joints[] = {-127, -74, -31, 32, 75, 128};
    const int slopesInv[] = {INTNN_MAX, 8, 2, 1, 2, 8, INTNN_MAX};  // Fixed slope inverses
    const int divisor = (1 << k) * numItems;

    /*printf("matIn in tahn:\n");
    intnn_print_mat(matIn);*/

    for (int r = 0; r < matOut->mRows; r++) {
        for (int c = 0; c < matOut->mCols; c++) {
            int x = intnn_get_elem(matIn, r, c) / divisor;
            int y, grad;
            
            if (x < joints[0])       { y = INTNN_MIN; grad = slopesInv[0]; }
            else if (x < joints[1])  { y = x/4 - 88; grad = slopesInv[1]; }
            else if (x < joints[2])  { y = x - 32; grad = slopesInv[2]; }
            else if (x < joints[3])  { y = 2*x; grad = slopesInv[3]; }
            else if (x < joints[4])  { y = x + 32; grad = slopesInv[4]; }
            else if (x < joints[5])  { y = x/4 + 88; grad = slopesInv[5]; }
            else                     { y = INTNN_MAX; grad = slopesInv[6]; }

            //printf("@%d, %d\n", x, y);
            
            intnn_set_elem(matOut, r, c, clamp(y, INTNN_MIN, INTNN_MAX));
            intnn_set_elem(matActvGradInv, r, c, grad);
        }
    }

    //printf("output in tanh:\n");
    //intnn_print_mat(matOut);
}

void intnn_rescale(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k) {
    const int divisor = 1 << k;
    intnn_set_all_constant(matActvGradInv, 1);
    
    for (int r = 0; r < intnn_rows(matOut); r++) {
        for (int c = 0; c < intnn_cols(matOut); c++) {
            int val = intnn_get_elem(matIn, r, c) / divisor;
            intnn_set_elem(matOut, r, c, val);
        }
    }
}

void intnn_softmax(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k) {
    for (int r = 0; r < intnn_rows(matOut); r++) {
        int rowSum = 0;
        // First pass: clamp negatives and compute row sum
        for (int c = 0; c < intnn_cols(matOut); c++) {
            int val = intnn_get_elem(matIn, r, c);
            if (val <= 0) {
                intnn_set_elem(matOut, r, c, 0);
            } else {
                intnn_set_elem(matOut, r, c, val);
                rowSum += val;
            }
        }

        rowSum = (rowSum == 0) ? 1 : rowSum;  // Avoid division by zero
        const int scaleFactor = INTNN_MAX / rowSum;
        
        // Second pass: rescale values
        for (int c = 0; c < intnn_cols(matOut); c++) {
            int current = intnn_get_elem(matOut, r, c);
            int newVal = (current == 0) ? 0 : current * scaleFactor;
            
            intnn_set_elem(matOut, r, c, newVal);
            intnn_set_elem(matActvGradInv, r, c, (current == 0) ? INTNN_MAX : 1);
        }
    }
}

void intnn_relu8bit(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k) {
    for (int r = 0; r < intnn_rows(matOut); r++) {
        for (int c = 0; c < intnn_cols(matOut); c++) {
            int val = intnn_get_elem(matIn, r, c);
            int clamped = clamp(val, 0, INTNN_MAX);
            int grad = (val < 0 || val > INTNN_MAX) ? INTNN_MAX : 1;
            
            intnn_set_elem(matOut, r, c, clamped);
            intnn_set_elem(matActvGradInv, r, c, grad);
        }
    }
}

void intnn_leakyrelu(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k) {
    const int leakSlope = 5;
    
    for (int r = 0; r < intnn_rows(matOut); r++) {
        for (int c = 0; c < intnn_cols(matOut); c++) {
            int val = intnn_get_elem(matIn, r, c);
            int y, grad;
            
            if (val < SHRT_MIN)      { y = SHRT_MIN; grad = INTNN_MAX; }
            else if (val < 0)        { y = val / leakSlope; grad = leakSlope; }
            else if (val < SHRT_MAX) { y = val; grad = 1; }
            else                     { y = SHRT_MAX; grad = INTNN_MAX; }
            
            intnn_set_elem(matOut, r, c, y);
            intnn_set_elem(matActvGradInv, r, c, grad);
        }
    }
}

void intnn_plu(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k) {
    const int slope = 10;  // Slope 1/a
    const int c = 1;
    
    for (int r = 0; r < intnn_rows(matOut); r++) {
        for (int c_idx = 0; c_idx < intnn_cols(matOut); c_idx++) {
            int x = intnn_get_elem(matIn, r, c_idx);
            int plu_min = (x - c) / slope + c;
            int plu_max = (x + c) / slope - c;
            int y = clamp(x, plu_min, plu_max);
            
            intnn_set_elem(matOut, r, c_idx, clamp(y, INTNN_MIN, INTNN_MAX));
            
            int grad = (x != 0) ? abs(y / x) : 1;
            intnn_set_elem(matActvGradInv, r, c_idx, grad);
        }
    }
}

void intnn_as_is(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k) {
    for (int r = 0; r < intnn_rows(matOut); r++) {
        for (int c = 0; c < intnn_cols(matOut); c++) {
            int val = intnn_get_elem(matIn, r, c);
            intnn_set_elem(matOut, r, c, val);
        }
    }
    intnn_set_all_constant(matActvGradInv, 1);
}

// 3D Activation Functions -----------------------------------------------------

void intnn_activate3d(intnn_mat3d* mat3dOut, intnn_mat3d* mat3dIn, 
                     intnn_mat3d* matActvGradInv3d, intnn_actv_type actv, 
                     int k, int numItems) {
    // Get dimensions for reuse
    const int depth = intnn_mat3d_depth(mat3dIn);
    const int rows = intnn_mat3d_rows(mat3dIn);
    const int cols = intnn_mat3d_cols(mat3dIn);
    
    // Reset output if dimensions mismatch
    if (!intnn_mat3d_dims_equal(mat3dOut, mat3dIn)) {
        intnn_reset_zero3d(mat3dOut, depth, rows, cols);
    }
    
    // Reset gradient tensor
    intnn_reset_zero3d(matActvGradInv3d, depth, rows, cols);

    // Process each depth slice
    for (int d = 0; d < depth; d++) {
        intnn_mat* outSlice = intnn_mat3d_get_mat_at_depth(mat3dOut, d);
        intnn_mat* inSlice = intnn_mat3d_get_mat_at_depth(mat3dIn, d);
        intnn_mat* gradSlice = intnn_mat3d_get_mat_at_depth(matActvGradInv3d, d);
        
        switch (actv) {
            case INTNN_ACTV_SIGMOID:
                intnn_sigmoid(outSlice, inSlice, gradSlice, k);
                break;
            case INTNN_ACTV_TANH:
                intnn_tanh(outSlice, inSlice, gradSlice, k, numItems);
                break;
            case INTNN_ACTV_RESCALE:
                intnn_rescale(outSlice, inSlice, gradSlice, k);
                break;
            default:
                printf("Unsupported 3D activation type\n");
                break;
        }
    }
}

void intnn_sigmoid3d(intnn_mat3d* mat3dOut, intnn_mat3d* mat3dIn, 
                     intnn_mat3d* matActvGradInv3d, int k) {
    const int depth = intnn_mat3d_depth(mat3dIn);
    for (int d = 0; d < depth; d++) {
        intnn_mat* outSlice = intnn_mat3d_get_mat_at_depth(mat3dOut, d);
        intnn_mat* inSlice = intnn_mat3d_get_mat_at_depth(mat3dIn, d);
        intnn_mat* gradSlice = intnn_mat3d_get_mat_at_depth(matActvGradInv3d, d);
        intnn_sigmoid(outSlice, inSlice, gradSlice, k);
    }
}

void intnn_tanh3d(intnn_mat3d* mat3dOut, intnn_mat3d* mat3dIn, 
                 intnn_mat3d* matActvGradInv3d, int k, int numItems) {
    const int depth = intnn_mat3d_depth(mat3dIn);
    for (int d = 0; d < depth; d++) {
        intnn_mat* outSlice = intnn_mat3d_get_mat_at_depth(mat3dOut, d);
        intnn_mat* inSlice = intnn_mat3d_get_mat_at_depth(mat3dIn, d);
        intnn_mat* gradSlice = intnn_mat3d_get_mat_at_depth(matActvGradInv3d, d);
        intnn_tanh(outSlice, inSlice, gradSlice, k, numItems);
    }
}

void intnn_rescale3d(intnn_mat3d* mat3dOut, intnn_mat3d* mat3dIn, 
                    intnn_mat3d* matActvGradInv3d, int k) {
    const int depth = intnn_mat3d_depth(mat3dIn);
    for (int d = 0; d < depth; d++) {
        intnn_mat* outSlice = intnn_mat3d_get_mat_at_depth(mat3dOut, d);
        intnn_mat* inSlice = intnn_mat3d_get_mat_at_depth(mat3dIn, d);
        intnn_mat* gradSlice = intnn_mat3d_get_mat_at_depth(matActvGradInv3d, d);
        intnn_rescale(outSlice, inSlice, gradSlice, k);
    }
}
