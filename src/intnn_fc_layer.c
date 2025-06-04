#include "intnn_fc_layer.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "intnn_actv.h"
#include "intnn_consts.h"
#include "intnn_loss.h"
#include "intnn_mat.h"
#include "intnn_mat3d.h"
#include "intnn_tools.h"
/**
 * @brief 创建一个全连接层（堆分配）
 */
intnn_fc_layer* intnn_fc_create(int inDim, int outDim) {
    // 在堆上分配一个层结构体
    intnn_fc_layer* layer = (intnn_fc_layer*)malloc(sizeof(intnn_fc_layer));
    if (!layer)
        return NULL;

    layer->mInDim = inDim;
    layer->mOutDim = outDim;
    layer->mInput = NULL;

    // 创建权重和偏置（由 intnn_create_mat 返回指针）
    layer->mWeight = intnn_create_mat(inDim, outDim);
    layer->mBias = intnn_create_mat(1, outDim);

    layer->mInter = NULL;
    layer->mOutput = NULL;

    layer->mDeltas = NULL;
    layer->mDeltasTranspose = NULL;
    layer->mDActvTranspose = NULL;
    layer->mActvGradInv = NULL;
    layer->mWeightUpdate = intnn_create_mat(inDim, outDim);
    layer->mBiasUpdate = intnn_create_mat(1, outDim);

    // BatchNorm 初始化为 NULL
    layer->mUseBn = false;
    layer->mMean = NULL;
    layer->mVariance = NULL;
    layer->mStdevWithEps = NULL;
    layer->mStandardized = NULL;
    layer->mGamma = NULL;
    layer->mBeta = NULL;
    layer->mBatchNormalized = NULL;
    layer->mDGamma = NULL;
    layer->mDBeta = NULL;
    layer->mDBn = NULL;
    layer->mGammaUpdate = NULL;
    layer->mBetaUpdate = NULL;

    // DFA 初始化为 NULL
    layer->mUseDfa = false;
    layer->mDfaWeight = NULL;

    // 默认激活
    layer->mActv = INTNN_ACTV_TANH;

    // 名称指针指向空，调用 set_name 可自行分配
    layer->mName = NULL;

    return layer;
}

/**
 * @brief 释放全连接层内部矩阵（不 free 本身指针）
 */
void intnn_fc_free(intnn_fc_layer* layer) {
    if (!layer)
        return;

    if (layer->mWeight)
        intnn_free_mat(layer->mWeight);
    if (layer->mBias)
        intnn_free_mat(layer->mBias);
    if (layer->mInter)
        intnn_free_mat(layer->mInter);
    if (layer->mOutput)
        intnn_free_mat(layer->mOutput);
    if (layer->mDeltas)
        intnn_free_mat(layer->mDeltas);
    if (layer->mDeltasTranspose)
        intnn_free_mat(layer->mDeltasTranspose);
    if (layer->mDActvTranspose)
        intnn_free_mat(layer->mDActvTranspose);
    if (layer->mActvGradInv)
        intnn_free_mat(layer->mActvGradInv);
    if (layer->mWeightUpdate)
        intnn_free_mat(layer->mWeightUpdate);
    if (layer->mBiasUpdate)
        intnn_free_mat(layer->mBiasUpdate);

    if (layer->mMean)
        intnn_free_mat(layer->mMean);
    if (layer->mVariance)
        intnn_free_mat(layer->mVariance);
    if (layer->mStdevWithEps)
        intnn_free_mat(layer->mStdevWithEps);
    if (layer->mStandardized)
        intnn_free_mat(layer->mStandardized);
    if (layer->mGamma)
        intnn_free_mat(layer->mGamma);
    if (layer->mBeta)
        intnn_free_mat(layer->mBeta);
    if (layer->mBatchNormalized)
        intnn_free_mat(layer->mBatchNormalized);
    if (layer->mDGamma)
        intnn_free_mat(layer->mDGamma);
    if (layer->mDBeta)
        intnn_free_mat(layer->mDBeta);
    if (layer->mDBn)
        intnn_free_mat(layer->mDBn);
    if (layer->mGammaUpdate)
        intnn_free_mat(layer->mGammaUpdate);
    if (layer->mBetaUpdate)
        intnn_free_mat(layer->mBetaUpdate);

    if (layer->mDfaWeight)
        intnn_free_mat(layer->mDfaWeight);

    if (layer->mName) {
        free(layer->mName);
        layer->mName = NULL;
    }
}

void intnn_fc_forward(intnn_fc_layer* layer, intnn_mat* x) {
    assert(layer != NULL && x != NULL);
    layer->mInput = x;

    // 1) 确保 mInter 已经分配
    if (!layer->mInter) {
        layer->mInter = intnn_create_mat(x->mRows, layer->mOutDim);
    }

    // 2) 确保 mOutput 和 mActvGradInv 已经分配
    if (!layer->mOutput) {
        layer->mOutput = intnn_create_mat(x->mRows, layer->mOutDim);
    }
    if (!layer->mActvGradInv) {
        layer->mActvGradInv = intnn_create_mat(x->mRows, layer->mOutDim);
    }

    // mInter = x · W
    intnn_mat_mul_mat(layer->mInter, x, layer->mWeight);

    if (layer->mUseBn) {
        // Calculate mean and variance (column-wise)
        intnn_average_colwise(layer->mMean);
        int avg = intnn_average(layer->mInter);
        intnn_variance_with_avg(layer->mVariance, avg);

        // stdev = sqrt(variance + epsilon)
        intnn_square_root_of(layer->mStdevWithEps, layer->mVariance);

        // standardized = (x - mean) / stdev
        intnn_standardize(layer->mStandardized, 3, SHRT_MIN, SHRT_MAX);

        // batchNorm = standardized * gamma + beta
        intnn_self_elem_mul_mat(layer->mBatchNormalized, layer->mGamma);
        intnn_self_add_mat(layer->mBatchNormalized, layer->mBeta);

        // activation(mBatchNormalized)
        intnn_activate(layer->mOutput, layer->mBatchNormalized,
                       layer->mActvGradInv, layer->mActv, layer->mOutput->mRows,
                       layer->mOutput->mCols);
    } else {
        // mInter += B
        intnn_self_add_mat(layer->mInter, layer->mBias);

        // 如果是“AS_IS”激活，需要让 mActvGradInv 全置为 1：
        if (layer->mActv == INTNN_ACTV_AS_IS) {
            // 把整个 (batchSize × outDim) 矩阵填 1
            intnn_set_all_constant(layer->mActvGradInv, 1);
        }

        // activation(mInter)
        intnn_activate(layer->mOutput, layer->mInter, layer->mActvGradInv,
                       layer->mActv, layer->mOutput->mRows,
                       layer->mOutput->mCols);
    }
}
void intnn_fc_backward(intnn_fc_layer* layer,
                       intnn_mat* lastDeltas,
                       int lrInv) {
    assert(layer != NULL && lastDeltas != NULL);
    if (lrInv <= 0)
        lrInv = 1;

    int batchSize = lastDeltas->mRows;
    int outDim = layer->mOutDim;
    int inDim = layer->mInDim;

    // 1. 确保 mDeltas 已分配：(batchSize, outDim)
    if (!layer->mDeltas) {
        layer->mDeltas = intnn_create_mat(batchSize, outDim);
    }

    // 2. 确保 mActvGradInv 已分配：(batchSize, outDim)
    if (!layer->mActvGradInv) {
        layer->mActvGradInv = intnn_create_mat(batchSize, outDim);
    }

    // 3. 计算 mDeltas = lastDeltas ⊙ mActvGradInv
    intnn_mat_elem_mul_mat(layer->mDeltas, lastDeltas, layer->mActvGradInv);


    // 4. 计算输入 X 的转置（shape = (inDim, batchSize)），用于后续权重更新
    //    用一个局部矩阵来存储，不要复用 mDeltasTranspose
    intnn_mat* inputT = intnn_create_mat(inDim, batchSize);
    intnn_transpose_of(inputT, layer->mInput);

    // 5. 计算 weightUpdate = inputT · mDeltas
    if (!layer->mWeightUpdate) {
        layer->mWeightUpdate = intnn_create_mat(inDim, outDim);
    }
    intnn_mat_mul_mat(layer->mWeightUpdate, inputT, layer->mDeltas);
    intnn_self_div_const(layer->mWeightUpdate, lrInv);
    intnn_self_sub_mat(layer->mWeight, layer->mWeightUpdate);

    // 6. 计算 biasUpdate = [1×N] · mDeltas
    //    —— 不要用局部 struct，而要动态创建一个矩阵
    intnn_mat* allOneRow = intnn_create_mat(1, batchSize);
    intnn_set_all_constant(allOneRow, 1);  // 填充全 1


    if (!layer->mBiasUpdate) {
        layer->mBiasUpdate = intnn_create_mat(1, outDim);
    }
    intnn_mat_mul_mat(layer->mBiasUpdate, allOneRow,
                      layer->mDeltas);  // (1, outDim)

    intnn_self_div_const(layer->mBiasUpdate, lrInv);
    intnn_self_sub_mat(layer->mBias, layer->mBiasUpdate);

    // 释放临时矩阵
    intnn_free_mat(allOneRow);


    // 7. 限制 weight, bias 范围
    intnn_clamp_mat(layer->mWeight, SHRT_MIN + 1, SHRT_MAX);
    intnn_clamp_mat(layer->mBias, SHRT_MIN + 1, SHRT_MAX);


    // 8. 计算 mDeltasTranspose = transpose(mDeltas)
    //    先分配 (outDim, batchSize)，再转置
    if (!layer->mDeltasTranspose) {
        layer->mDeltasTranspose = intnn_create_mat(outDim, batchSize);
    }
    intnn_transpose_of(layer->mDeltasTranspose, layer->mDeltas);


    // 9. 释放掉临时的 inputT
    intnn_free_mat(inputT);
}

intnn_mat* intnn_fc_get_output(intnn_fc_layer* layer) {
    assert(layer != NULL);
    return layer->mOutput;
}

intnn_mat* intnn_fc_get_weight(intnn_fc_layer* layer) {
    assert(layer != NULL);
    return layer->mWeight;
}

intnn_mat* intnn_fc_get_deltas_transpose(intnn_fc_layer* layer) {
    assert(layer != NULL);
    return layer->mDeltasTranspose;
}

void intnn_fc_set_name(intnn_fc_layer* layer, const char* name) {
    strncpy(layer->mName, name, sizeof(layer->mName));
    layer->mName[sizeof(layer->mName) - 1] = '\\0';
}

void intnn_fc_set_actv(intnn_fc_layer* layer, intnn_actv_type actv) {
    layer->mActv = actv;
}

void intnn_fc_set_random_weight_bias(intnn_fc_layer* layer) {
    intnn_set_random(layer->mWeight, false, -127, 127);
    intnn_set_random(layer->mBias, true, 0, 0);
}

void intnn_fc_set_he_init(intnn_fc_layer* layer) {
    int range = sqrt((12 * INTNN_MAX) / (layer->mInDim + layer->mOutDim));
    intnn_set_random(layer->mWeight, false, -range, range);
    intnn_set_random(layer->mBias, false, -range, range);
}

void intnn_fc_use_bn(intnn_fc_layer* layer, bool use_bn) {
    layer->mUseBn = use_bn;
}

void intnn_fc_use_dfa(intnn_fc_layer* layer, bool use_dfa) {
    layer->mUseDfa = use_dfa;
    if (use_dfa) {
        intnn_set_all_constant(layer->mWeight, 0);
        intnn_set_all_constant(layer->mBias, 0);
    }
}

void intnn_fc_set_random_weight(intnn_fc_layer* layer) {
    assert(layer && layer->mWeight);
    intnn_set_random(layer->mWeight, false, -127, 127);  // 使用已有的随机函数
}

void intnn_fc_set_random_bias(intnn_fc_layer* layer) {
    assert(layer && layer->mBias);
    intnn_set_random(layer->mBias, false, -127, 127);  // 使用已有的随机函数
}

void intnn_fc_set_random_dfa_weight(intnn_fc_layer* layer, int rows, int cols) {
    if (!layer)
        return;
    if (layer->mDfaWeight) {
        intnn_free_mat(layer->mDfaWeight);  // 如果已有旧值，释放
    }
    layer->mDfaWeight = intnn_create_mat(rows, cols);
    intnn_set_random(layer->mDfaWeight, false, -127,
                     127);  // 使用已有的随机函数
}
void intnn_fc_init_he_weight_bias(intnn_fc_layer* layer) {
    assert(layer && layer->mWeight && layer->mBias);

    // Calculate range for He initialization
    int range = sqrt((12 * INTNN_MAX) / (layer->mInDim + layer->mOutDim));

    // Initialize weights and biases
    intnn_set_random(layer->mWeight, false, -range, range);
    intnn_set_all_constant(layer->mBias,
                           0);  // He initialization typically sets bias to 0
}

void intnn_fc_use_batch_normalization(intnn_fc_layer* layer, bool useBn) {
    if (!layer)
        return;
    layer->mUseBn = useBn;
}

void intnn_fc_print_weight(intnn_fc_layer* layer, FILE* out) {
    if (!layer || !layer->mWeight)
        return;
    intnn_print_mat(layer->mWeight);
}

void intnn_fc_print_bias(intnn_fc_layer* layer, FILE* out) {
    if (!layer || !layer->mBias)
        return;
    intnn_print_mat(layer->mBias);
}

void intnn_fc_print_inter(intnn_fc_layer* layer, FILE* out) {
    if (!layer || !layer->mInter)
        return;
    intnn_print_mat(layer->mInter);
}

void intnn_fc_print_output(intnn_fc_layer* layer, FILE* out) {
    if (!layer || !layer->mOutput)
        return;
    intnn_print_mat(layer->mOutput);
}
