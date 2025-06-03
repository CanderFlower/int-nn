#ifndef INTNN_LOSS_H
#define INTNN_LOSS_H

#include "intnn_mat.h"
#include "intnn_tools.h"

// 回归相关
int intnn_scalar_l2_loss(int y, int y_hat);                      // 单个值的 L2 损失
int intnn_scalar_l2_loss_delta(int y, int y_hat);                // 单个值的 L2 导数
int intnn_batch_l2_loss(intnn_mat* loss_mat, const intnn_mat* y_mat, const intnn_mat* y_hat_mat);         // 批量 L2 损失
int intnn_batch_l2_loss_delta(intnn_mat* delta_mat, const intnn_mat* y_mat, const intnn_mat* y_hat_mat);  // 批量 L2 导数

// 分类相关（口袋交叉熵等）
int intnn_vector_pocket_cross_loss(intnn_mat* loss_vec, const intnn_mat* y_vec, const intnn_mat* y_hat_vec);  // 向量交叉熵
int intnn_batch_pocket_cross_loss(intnn_mat* loss_mat, const intnn_mat* y_mat, const intnn_mat* y_hat_mat);  // 批量口袋交叉熵
int intnn_batch_pocket_cross_loss_delta(intnn_mat* delta_mat, const intnn_mat* y_mat, const intnn_mat* y_hat_mat); // 批量口袋交叉熵导数

int intnn_batch_cross_entropy_loss(intnn_mat* loss_mat, const intnn_mat* y_mat, const intnn_mat* y_hat_mat);       // 真实交叉熵损失
int intnn_batch_cross_entropy_loss_delta(intnn_mat* delta_mat, const intnn_mat* y_mat, const intnn_mat* y_hat_mat); // 真实交叉熵导数

#endif // INTNN_LOSS_H
