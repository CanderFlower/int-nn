#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "intnn_examples.h"
#include "intnn_fc_layer.h"
#include "intnn_mat.h"
#include "intnn_consts.h"
#include "intnn_actv.h"
#include "intnn_tools.h"

int example_intnn_fc_dfa_mnist() {
    const int numTrain = 60000;
    const int numTest = 10000;
    const int numClasses = 10;
    const int dimInput = 28 * 28;
    const int dim1 = 100;
    const int dim2 = 50;
    const int epochs = 5;
    const int miniBatchSize = 20;
    int lrInv = 1000;

    srand(114514);

    // 加载数据
    intnn_mat* trainImages = intnn_create_mat(numTrain, dimInput);
    intnn_mat* trainLabels = intnn_create_mat(numTrain, 1);
    intnn_mat* testImages = intnn_create_mat(numTest, dimInput);
    intnn_mat* testLabels = intnn_create_mat(numTest, 1);
    intnn_load_mnist_images(trainImages, numTrain, true);
    intnn_load_mnist_labels(trainLabels, numTrain, true);
    intnn_load_mnist_images(testImages, numTest, false);
    intnn_load_mnist_labels(testLabels, numTest, false);
    printf("Loaded MNIST train/test samples.\n");

    // 构造 One-hot 目标
    intnn_mat* trainTarget = intnn_create_mat(numTrain, numClasses);
    intnn_mat* testTarget = intnn_create_mat(numTest, numClasses);
    for (int i = 0; i < numTrain; ++i)
        trainTarget->mMat[i][trainLabels->mMat[i][0]] = INTNN_UNSIGNED_4BIT_MAX;
    for (int i = 0; i < numTest; ++i)
        testTarget->mMat[i][testLabels->mMat[i][0]] = INTNN_UNSIGNED_4BIT_MAX;

    // 创建训练用层
    intnn_fc_layer* fc1 = intnn_fc_create(dimInput, dim1);
    intnn_fc_layer* fc2 = intnn_fc_create(dim1, dim2);
    intnn_fc_layer* fc3 = intnn_fc_create(dim2, numClasses);
    intnn_fc_set_actv(fc1, INTNN_ACTV_TANH);
    intnn_fc_set_actv(fc2, INTNN_ACTV_TANH);
    intnn_fc_set_actv(fc3, INTNN_ACTV_TANH);
    intnn_fc_use_dfa(fc1, true);
    intnn_fc_use_dfa(fc2, true);
    intnn_fc_use_dfa(fc3, true);

	fc1->mWeight = intnn_create_mat(dimInput, dim1);
	fc2->mWeight = intnn_create_mat(dim1, dim2);
	fc3->mWeight = intnn_create_mat(dim2, numClasses);
	fc1->mBias = intnn_create_mat(1, dim1);
	fc2->mBias = intnn_create_mat(1, dim2);
	fc3->mBias = intnn_create_mat(1, numClasses);

    fc1->mNext = fc2;
    fc2->mPrev = fc1;
    fc2->mNext = fc3;
    fc3->mPrev = fc2;
    fc3->mNext = NULL; // 最后一层没有下一层
    fc1->mPrev = NULL; // 第一层没有前一层

    int correct;

    //// 初始化前向精度（训练用）
    intnn_fc_forward(fc1, trainImages);
    correct = intnn_count_max_match(intnn_fc_get_output(fc3), trainTarget);
    printf("Initial training correct: %d / %d\n", correct, numTrain);
    printf("Initial training accuracy: %.2f%%\n", correct * 100.0 / numTrain);

    intnn_fc_forward(fc1, testImages);
    correct = intnn_count_max_match(intnn_fc_get_output(fc3), testTarget);
    printf("Initial test correct: %d / %d\n", correct, numTest);
    printf("Initial test accuracy: %.2f%%\n", correct * 100.0 / numTest);

    // 训练过程
    int* indices = malloc(sizeof(int) * numTrain);
    for (int i = 0; i < numTrain; ++i) indices[i] = i;

    intnn_mat* miniX = intnn_create_mat(miniBatchSize, dimInput);
    intnn_mat* miniY = intnn_create_mat(miniBatchSize, numClasses);
    intnn_mat* lossMat = intnn_create_mat(miniBatchSize, numClasses);
    intnn_mat* deltaMat = intnn_create_mat(miniBatchSize, numClasses);
    printf("Epoch,\tTrainLoss,\tTrainAcc,\tTestAcc\n");

    clock_t start = clock();

    for (int ep = 1; ep <= epochs; ++ep) {
        intnn_tools_shuffle_indices(indices, numTrain);
        int totalCorrect = 0;
        int totalLoss = 0;

        for (int i = 0; i < numTrain / miniBatchSize; ++i) {
            intnn_indexed_slice_of(miniX, trainImages, indices, i * miniBatchSize, (i + 1) * miniBatchSize);

           /* printf("\n======================================\n");
            printf("FORWARD START:\n");
            printf("\n======================================\n");*/
            intnn_fc_forward(fc1, miniX);
            int aa = 0;
            intnn_indexed_slice_of(miniY, trainTarget, indices, i * miniBatchSize, (i + 1) * miniBatchSize);
            totalLoss += intnn_batch_l2_loss(lossMat, miniY, intnn_fc_get_output(fc3));
            intnn_batch_l2_loss_delta(deltaMat, miniY, intnn_fc_get_output(fc3));
            totalCorrect += intnn_count_max_match(intnn_fc_get_output(fc3), miniY);

            /*printf("\n======================================\n");
            printf("BACKWARD START:\n");
            printf("\n======================================\n");*/
            intnn_fc_backward(fc3, deltaMat, lrInv);
        }

        intnn_fc_forward(fc1, testImages);
        int testCorrect = intnn_count_max_match(intnn_fc_get_output(fc3), testTarget);

        printf("%d,\t%-8d,\t%.2f%%,\t\t%.2f%%\n", ep, totalLoss,
            totalCorrect * 100.0 / numTrain,
            testCorrect * 100.0 / numTest);

        if(ep == epochs){
            printf("Final training accuracy: %.2f%%\n", totalCorrect * 100.0 / numTrain);
            printf("Final test accuracy: %.2f%%\n", testCorrect * 100.0 / numTest);
        }

        if ((ep % 10 == 0) && lrInv < 20000) lrInv *= 2;
    }

    clock_t end = clock();
    double elapsed_secs = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Training time: %.2f seconds\n", elapsed_secs);

    // 释放所有资源
    intnn_fc_free(fc1);
    intnn_fc_free(fc2);
    intnn_fc_free(fc3);
    intnn_free_mat(trainImages); intnn_free_mat(trainLabels);
    intnn_free_mat(testImages);  intnn_free_mat(testLabels);
    intnn_free_mat(trainTarget); intnn_free_mat(testTarget);
    intnn_free_mat(miniX);       intnn_free_mat(miniY);
    intnn_free_mat(lossMat);     intnn_free_mat(deltaMat);
    free(indices);
    return 0;
}
