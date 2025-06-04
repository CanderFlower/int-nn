#ifndef INTNN_ACTV_H
#define INTNN_ACTV_H

#include "intnn_mat.h"
#include "intnn_mat3d.h"
#include "intnn_consts.h"

#ifdef __cplusplus
extern "C" {
#endif

// 激活函数类型枚举
typedef enum {
    INTNN_ACTV_SIGMOID,
    INTNN_ACTV_TANH,
    INTNN_ACTV_RESCALE,
    INTNN_ACTV_SOFTMAX,
    INTNN_ACTV_RELU8BIT,
    INTNN_ACTV_LEAKYRELU,
    INTNN_ACTV_PLU,
    INTNN_ACTV_AS_IS
} intnn_actv_type;

// 主激活函数（2D）
void intnn_activate(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv,
                    intnn_actv_type actv, int k, int numItems);

// 主激活函数（3D）
void intnn_activate3d(intnn_mat3d* mat3dOut, intnn_mat3d* mat3dIn, intnn_mat3d* matActvGradInv,
                      intnn_actv_type actv, int k, int numItems);

// 单个激活函数实现（2D）
void intnn_sigmoid(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k);
void intnn_tanh(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k, int numItems);
void intnn_rescale(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k);
void intnn_softmax(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k);
void intnn_relu8bit(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k);
void intnn_leakyrelu(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k);
void intnn_plu(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k);
void intnn_as_is(intnn_mat* matOut, intnn_mat* matIn, intnn_mat* matActvGradInv, int k);

// 单个激活函数实现（3D）
void intnn_sigmoid3d(intnn_mat3d* matOut, intnn_mat3d* matIn, intnn_mat3d* matActvGradInv, int k);
void intnn_tanh3d(intnn_mat3d* matOut, intnn_mat3d* matIn, intnn_mat3d* matActvGradInv, int k, int numItems);
void intnn_rescale3d(intnn_mat3d* matOut, intnn_mat3d* matIn, intnn_mat3d* matActvGradInv, int k);

#ifdef __cplusplus
}
#endif

#endif // INTNN_ACTV_H
