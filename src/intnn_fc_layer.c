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

	layer->mNext = NULL; // 下一层指针
	layer->mPrev = NULL; // 前一层指针

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
    
    layer->mInput = intnn_copy_mat(x);

    if (layer->mInter) intnn_free_mat(layer->mInter);
    layer->mInter = intnn_create_mat(x->mRows, layer->mWeight->mCols);
    intnn_mat_mul_mat(layer->mInter, x, layer->mWeight); // (N, D(k)) = (N, D(k-1)) × (D(k-1), D(k)) 

    //printf("X OF (%d, %d):\n", layer->mInDim, layer->mOutDim);
    //intnn_print_mat(x);
   /* for (int i = 0; i < 10; i++, printf("\n"))
        for (int j = 0; j < 10; j++)
            printf("%d ", x->mMat[i][j]);*/
    //printf("W OF (%d, %d):\n", layer->mInDim, layer->mOutDim);
    /*for (int i = 0; i < 10; i++, printf("\n"))
        for (int j = 0; j < 10; j++)
            printf("%d ", layer->mWeight->mMat[i][j]);*/
    /*intnn_print_mat(layer->mWeight);
    
    printf("INTER OF (%d, %d):\n", layer->mInDim, layer->mOutDim);
    intnn_print_mat(layer->mInter);*/
    /*for(int i=0;i<10;i++,printf("\n"))
        for(int j=0;j<10;j++)
			printf("%d ", layer->mInter->mMat[i][j]);*/

    if(layer->mUseBn){
        assert(0); // 不支持
    }else{
        intnn_self_add_mat(layer->mInter, layer->mBias); // (N, D(k)) += (1, D(k)) => (N, D(k))

        if (layer->mOutput) intnn_free_mat(layer->mOutput);
        layer->mOutput = intnn_create_mat(layer->mInter->mRows, layer->mInter->mCols);

        if (layer->mActvGradInv) intnn_free_mat(layer->mActvGradInv);
        layer->mActvGradInv = intnn_create_mat(layer->mInter->mRows, layer->mInter->mCols);

        intnn_activate(layer->mOutput, layer->mInter, layer->mActvGradInv,
            layer->mActv, INTNN_K_BIT, layer->mInDim); // (N, D(k)) = activation((N, D(k)))
    }


    if(layer->mNext != NULL){
        intnn_fc_forward(layer->mNext, layer->mOutput); // 递归调用下一层
    }
}

void intnn_fc_backward(intnn_fc_layer* layer,
    intnn_mat* lastDeltas,
    int lrInv) {
     // COMPUTE DELTAS
    
    if(layer->mUseBn) {
        assert(0); // 不支持
    }

    if(layer->mNext == NULL){
        if (layer->mDeltas) intnn_free_mat(layer->mDeltas);
        layer->mDeltas = intnn_create_mat(lastDeltas->mRows, lastDeltas->mCols);

        intnn_mat_elem_div_mat(layer->mDeltas, lastDeltas, layer->mActvGradInv); // (N, D(k)) = (N, D(k)) / (1, D(k))
    }
    else {
        if (!layer->mUseDfa) {
            assert(0); // 不支持
        }
        else {
            if (!layer->mDfaWeight) {
                int range = intnn_floor_sqrt((12 * SHRT_MAX) / (layer->mInDim + layer->mOutDim));
                layer->mDfaWeight = intnn_create_mat(lastDeltas->mCols, layer->mWeight->mCols);
                intnn_set_random(layer->mDfaWeight, false, -range, range);
                //printf("DFA initialized!\n");
                //printf("initial DFA of layer %d->%d, size:(%d, %d)\n", layer->mInDim, layer->mOutDim, layer->mDfaWeight->mRows, layer->mDfaWeight->mCols);
                //intnn_print_mat(layer->mDfaWeight);
            }
            if (layer->mDeltas) intnn_free_mat(layer->mDeltas);
            layer->mDeltas = intnn_create_mat(lastDeltas->mRows, layer->mDfaWeight->mCols);
            intnn_mat_mul_mat(layer->mDeltas, lastDeltas, layer->mDfaWeight); // (N, D(k)) = (N, D(k-1)) × (D(k-1), D(k))
            intnn_self_elem_div_mat(layer->mDeltas, layer->mActvGradInv); // (N, D(k)) = (N, D(k)) / (1, D(k))
        }
    }
    
    if (layer->mDeltasTranspose) intnn_free_mat(layer->mDeltasTranspose);
    layer->mDeltasTranspose = intnn_create_mat(layer->mDeltas->mCols, layer->mDeltas->mRows);

    intnn_transpose_of(layer->mDeltasTranspose, layer->mDeltas); // (D(k), N) = (N, D(k))

    // AFTER COMPUTE DELTAS

    /*printf("MDELTAS of layer %d->%d \n", layer->mInDim, layer->mOutDim);
    for(int i=0;i<10;i++,printf("\n"))
		for (int j = 0; j < 10; j++)
			printf("%d | ", layer->mDeltas->mMat[i][j]);

    printf("LAST LAYER DELTAS of layer %d->%d:\n", layer->mInDim, layer->mOutDim);
    for (int i = 0; i < 10; i++, printf("\n"))
        for (int j = 0; j < 10; j++)
            printf("%d | ", lastDeltas->mMat[i][j]);*/
    

    int batchSize = layer->mDeltas->mRows;

    intnn_mat* prevOutputTranspose;
    if(layer->mPrev != NULL) {
        prevOutputTranspose = intnn_create_mat(layer->mPrev->mOutput->mCols, layer->mPrev->mOutput->mRows);
        intnn_transpose_of(prevOutputTranspose, layer->mPrev->mOutput); // (D(k-1), N) = (N, D(k-1))
    } else {
        prevOutputTranspose = intnn_create_mat(layer->mInput->mCols, layer->mInput->mRows);
        intnn_transpose_of(prevOutputTranspose, layer->mInput); // (D(k-1), N) = (N, D(k-1))
    }

    //intnn_print_mat(layer->mDeltas);

    if (!layer->mWeightUpdate) layer->mWeightUpdate = intnn_create_mat(layer->mInDim, layer->mOutDim);
    intnn_reset_zero(layer->mWeightUpdate, layer->mInDim, layer->mOutDim); // 重置权重更新矩阵
    intnn_mat_mul_mat(layer->mWeightUpdate, prevOutputTranspose, layer->mDeltas); // (D(k-1), D(k)) = (D(k-1), N) × (N, D(k))

    intnn_self_div_const(layer->mWeightUpdate, -lrInv); // (D(k-1), D(k)) /= -lrInv


    /*printf("WEIGHT UPDATE of layer %d->%d:\n", layer->mInDim, layer->mOutDim);
    for (int i = 0; i < layer->mWeightUpdate->mRows; i++, printf("\n"))
        for (int j = 0; j < layer->mWeightUpdate->mCols; j++)
            printf("%d | ", layer->mWeightUpdate->mMat[i][j]);*/

    intnn_self_add_mat(layer->mWeight, layer->mWeightUpdate); // (D(k-1), D(k)) += (D(k-1), D(k))

    //intnn_print_mat(layer->mWeightUpdate);

    if(layer->mUseBn){
        assert(0);
    }else{
        intnn_mat* allOneMat = intnn_create_mat(1, batchSize);
        intnn_reset_all_ones(allOneMat, 1, batchSize); // (1, N) = (1, N)

        if (layer->mBiasUpdate) intnn_free_mat(layer->mBiasUpdate);
        layer->mBiasUpdate = intnn_create_mat(allOneMat->mRows, layer->mDeltas->mCols);
        intnn_mat_mul_mat(layer->mBiasUpdate, allOneMat, layer->mDeltas); // (1, D(k)) = (1, N) × (N, D(k))
        intnn_self_div_const(layer->mBiasUpdate, -lrInv); // (1, D(k)) /= -lrInv

        if (!layer->mBias) layer->mBias = intnn_create_mat(layer->mBiasUpdate->mRows, layer->mBiasUpdate->mCols);
        intnn_self_add_mat(layer->mBias, layer->mBiasUpdate); // (1, D(k)) += (1, D(k))
        intnn_free_mat(allOneMat); // 释放临时矩阵
    }

	/*printf("Size: %d, %d\n", layer->mWeight->mRows, layer->mWeight->mCols);
    printf("Weight:\n");
    for (int i = 0; i < 10; i++, printf("\n"))
        for (int j = 0; j < 10; j++)
            printf("%d ", layer->mWeight->mMat[i][j]);
    printf("Bias:\n");
	for (int i = 0; i < 10; i++, printf("\n"))
		printf("%d ", layer->mBias->mMat[0][i]);*/

    intnn_free_mat(prevOutputTranspose); // 释放转置矩阵

    intnn_clamp_mat(layer->mWeight, -32767, 32767); // 限制权重范围
    intnn_clamp_mat(layer->mBias, -32767, 32767); // 限制偏置范围

    if(layer->mPrev != NULL){
        intnn_fc_backward(layer->mPrev, lastDeltas, lrInv); // 递归调用上一层
    }
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
    layer->mName[sizeof(layer->mName) - 1] = '\0';
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

void intnn_fc_copy_weights(intnn_fc_layer* dest, const intnn_fc_layer* src) {
    dest->mWeight = intnn_copy_mat(src->mWeight);
    dest->mBias = intnn_copy_mat(src->mBias);
}
