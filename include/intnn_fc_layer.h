#ifndef INTNN_FC_LAYER_H
#define INTNN_FC_LAYER_H

#include <stdio.h>
#include <stdbool.h>
#include "intnn_mat.h"
#include "intnn_actv.h"
#include "intnn_tools.h"
#include "intnn_consts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int mInDim;
    int mOutDim;

    // Input pointer (not owned)
    intnn_mat* mInput;

    // Weights and bias
    intnn_mat* mWeight;       // shape: (mInDim, mOutDim)
    intnn_mat* mBias;         // shape: (1, mOutDim)

    // Intermediate (pre-activation) and output
    intnn_mat* mInter;        // shape: (batchSize, mOutDim)
    intnn_mat* mOutput;       // shape: (batchSize, mOutDim)

    // Deltas and gradients for backward
    intnn_mat* mDeltas;            // shape: (batchSize, mOutDim)
    intnn_mat* mDeltasTranspose;   // shape: (mOutDim, batchSize)
    intnn_mat* mDActvTranspose;    // shape: (mOutDim, batchSize)
    intnn_mat* mActvGradInv;       // shape: (batchSize, mOutDim)
    intnn_mat* mWeightUpdate;      // shape: (mInDim, mOutDim)
    intnn_mat* mBiasUpdate;        // shape: (1, mOutDim)

    // Batch normalization parameters
    bool mUseBn;
    intnn_mat* mMean;              // shape: (1, mOutDim)
    intnn_mat* mVariance;          // shape: (1, mOutDim)
    intnn_mat* mStdevWithEps;      // shape: (1, mOutDim)
    intnn_mat* mStandardized;      // shape: (batchSize, mOutDim)
    intnn_mat* mGamma;             // shape: (1, mOutDim)
    intnn_mat* mBeta;              // shape: (1, mOutDim)
    intnn_mat* mBatchNormalized;   // shape: (batchSize, mOutDim)
    intnn_mat* mDGamma;            // shape: (1, mOutDim)
    intnn_mat* mDBeta;             // shape: (1, mOutDim)
    intnn_mat* mDBn;               // shape: (batchSize, mOutDim)
    intnn_mat* mGammaUpdate;       // shape: (1, mOutDim)
    intnn_mat* mBetaUpdate;        // shape: (1, mOutDim)

    // Direct Feedback Alignment (DFA)
    bool mUseDfa;
    intnn_mat* mDfaWeight;         // shape: (mOutDim, ?), user-defined

    // Activation type
    intnn_actv_type mActv;

    // Layer name
    char* mName;

} intnn_fc_layer;

/**
 * @brief 创建一个全连接层
 * 
 * @param inDim   输入维度
 * @param outDim  输出维度
 * @return intnn_fc_layer* 已分配并初始化的层，失败返回 NULL
 */
intnn_fc_layer* intnn_fc_create(int inDim, int outDim);

/**
 * @brief 释放全连接层以及其所有内部矩阵
 * 
 * @param layer  要释放的层
 */
void intnn_fc_free(intnn_fc_layer* layer);

/**
 * @brief 前向传播：给定输入 X，计算输出 Y = activation(X·W + B)
 * 
 * @param layer    全连接层
 * @param x        输入矩阵，形状为 (batchSize, inDim)
 *                 注意：此函数不会修改 x，本层持有指针 mInput
 *                 计算结果存放在 layer->mOutput
 */
void intnn_fc_forward(intnn_fc_layer* layer, intnn_mat* x);

/**
 * @brief 反向传播：给定上一层的误差 lastDeltas，计算本层权重和偏置更新／下游误差
 * 
 * @param layer         全连接层
 * @param lastDeltas    上一层（或损失层）传回的误差，形状 (batchSize, outDim)
 * @param lrInv         学习率的倒数（例如 lrInv = 1/learning_rate）
 *                      若 lrInv <= 0，则认为 lrInv = 1
 */
void intnn_fc_backward(intnn_fc_layer* layer, intnn_mat* lastDeltas, int lrInv);

/**
 * @brief 获取本层输出的指针（用于 FC 后续层直接使用）
 * 
 * @param layer  全连接层
 * @return intnn_mat* 输出矩阵，形状 (batchSize, outDim)
 */
intnn_mat* intnn_fc_get_output(intnn_fc_layer* layer);

/**
 * @brief 获取本层权重矩阵指针
 * 
 * @param layer  全连接层
 * @return intnn_mat* 权重矩阵，形状 (inDim, outDim)
 */
intnn_mat* intnn_fc_get_weight(intnn_fc_layer* layer);

/**
 * @brief 获取本层误差转置矩阵（误差维度为 (batchSize, outDim)，转置为 (outDim, batchSize)）
 * 
 * @param layer  全连接层
 * @return intnn_mat* 误差转置矩阵
 */
intnn_mat* intnn_fc_get_deltas_transpose(intnn_fc_layer* layer);

/**
 * @brief 设置层的名称（内部会复制一份字符串）
 * 
 * @param layer  全连接层
 * @param name   层名称
 */
void intnn_fc_set_name(intnn_fc_layer* layer, const char* name);

/**
 * @brief 随机初始化权重（标准正态分布或均匀分布）
 * 
 * @param layer  全连接层
 */
void intnn_fc_set_random_weight(intnn_fc_layer* layer);

/**
 * @brief 随机初始化偏置
 * 
 * @param layer  全连接层
 */
void intnn_fc_set_random_bias(intnn_fc_layer* layer);

/**
 * @brief 随机初始化 DFA 权重矩阵，形状为 (outDim, ???)
 * 
 * @param layer  全连接层
 * @param rows   行数
 * @param cols   列数
 */
void intnn_fc_set_random_dfa_weight(intnn_fc_layer* layer, int rows, int cols);

/**
 * @brief 设置激活函数类型
 * 
 * @param layer   全连接层
 * @param actv    激活函数枚举
 */
void intnn_fc_set_actv(intnn_fc_layer* layer, intnn_actv_type actv);

/**
 * @brief 使用 He 初始化权重和偏置
 * 
 * @param layer  全连接层
 */
void intnn_fc_init_he_weight_bias(intnn_fc_layer* layer);

/**
 * @brief 是否启用批归一化
 * 
 * @param layer  全连接层
 * @param useBn  true 启用，false 不启用
 */
void intnn_fc_use_batch_normalization(intnn_fc_layer* layer, bool useBn);

/**
 * @brief 是否启用直接反馈对齐（DFA）
 * 
 * @param layer   全连接层
 * @param useDfa  true 启用，false 不启用
 */
void intnn_fc_use_dfa(intnn_fc_layer* layer, bool useDfa);

/**
 * @brief 打印权重矩阵到指定输出（例如 stdout 或文件）
 * 
 * @param layer  全连接层
 * @param out    输出文件指针，若为 NULL 则使用 stdout
 */
void intnn_fc_print_weight(intnn_fc_layer* layer, FILE* out);

/**
 * @brief 打印偏置矩阵
 * 
 * @param layer  全连接层
 * @param out    输出文件指针
 */
void intnn_fc_print_bias(intnn_fc_layer* layer, FILE* out);

/**
 * @brief 打印中间结果（mInter）
 * 
 * @param layer  全连接层
 * @param out    输出文件指针
 */
void intnn_fc_print_inter(intnn_fc_layer* layer, FILE* out);

/**
 * @brief 打印输出结果（mOutput）
 * 
 * @param layer  全连接层
 * @param out    输出文件指针
 */
void intnn_fc_print_output(intnn_fc_layer* layer, FILE* out);

#ifdef __cplusplus
}
#endif

#endif // INTNN_FC_LAYER_H
