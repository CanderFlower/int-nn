#include "intnn_loss.h"
#include <assert.h>

int intnn_scalar_l2_loss(int y, int yHat) {
    int diff = yHat - y;
    return (diff * diff) / 2;
}

int intnn_scalar_l2_loss_delta(int y, int yHat) {
    return yHat - y;
}

int intnn_batch_l2_loss(intnn_mat* lossMat, const intnn_mat* yMat, const intnn_mat* yHatMat) {
    assert(intnn_dims_equal(yMat, yHatMat));

    int rows = intnn_rows(yHatMat);
    int cols = intnn_cols(yHatMat);
    intnn_reset_zero(lossMat, rows, 1);

    int accumLoss = 0;
    for (int r = 0; r < rows; ++r) {
        int rowLoss = 0;
        for (int c = 0; c < cols; ++c) {
            int y = intnn_get_elem(yMat, r, c);
            int yHat = intnn_get_elem(yHatMat, r, c);
            rowLoss += intnn_scalar_l2_loss(y, yHat);
        }
        intnn_set_elem(lossMat, r, 0, rowLoss);
        accumLoss += rowLoss;
    }

    return accumLoss;
}

int intnn_batch_l2_loss_delta(intnn_mat* lossDeltaMat, const intnn_mat* yMat, const intnn_mat* yHatMat) {
    assert(intnn_dims_equal(yMat, yHatMat));

    int rows = intnn_rows(yHatMat);
    int cols = intnn_cols(yHatMat);
    intnn_reset_zero(lossDeltaMat, rows, cols);

    int accumDelta = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int y = intnn_get_elem(yMat, r, c);
            int yHat = intnn_get_elem(yHatMat, r, c);
            int delta = intnn_scalar_l2_loss_delta(y, yHat);
            intnn_set_elem(lossDeltaMat, r, c, delta);
            accumDelta += delta;
        }
    }

    return accumDelta;
}

int intnn_vector_pocket_cross_loss(intnn_mat* loss_vec, const intnn_mat* y_vec, const intnn_mat* y_hat_vec) {
    assert(intnn_dims_equal(y_vec, y_hat_vec));
    assert(y_vec->mRows == 1);

    intnn_reset_zero(loss_vec, 1, 1);

    int sumLoss = 0;
    for (int c = 0; c < y_vec->mCols; ++c) {
        if (intnn_get_elem(y_vec, 0, c) == INT_MAX) {
            int loss = INT_MAX - intnn_get_elem(y_hat_vec, 0, c);
            intnn_set_elem(loss_vec, 0, 0, loss);
            sumLoss += loss;
        }
    }
    return sumLoss;
}
int intnn_batch_pocket_cross_loss(intnn_mat* loss_mat, const intnn_mat* y_mat, const intnn_mat* y_hat_mat) {
    assert(intnn_dims_equal(y_mat, y_hat_mat));

    intnn_reset_zero(loss_mat, y_mat->mRows, 1);

    int sumLoss = 0;
    for (int r = 0; r < y_mat->mRows; ++r) {
        for (int c = 0; c < y_mat->mCols; ++c) {
            if (intnn_get_elem(y_mat, r, c) == INT_MAX) {
                int loss = INT_MAX - intnn_get_elem(y_hat_mat, r, c);
                intnn_set_elem(loss_mat, r, 0, loss);
                sumLoss += loss;
            }
        }
    }
    return sumLoss;
}
int intnn_batch_pocket_cross_loss_delta(intnn_mat* delta_mat, const intnn_mat* y_mat, const intnn_mat* y_hat_mat) {
    assert(intnn_dims_equal(y_mat, y_hat_mat));

    intnn_reset_zero(delta_mat, y_hat_mat->mRows, y_hat_mat->mCols);

    int sumDelta = 0;
    for (int r = 0; r < y_mat->mRows; ++r) {
        for (int c = 0; c < y_mat->mCols; ++c) {
            if (intnn_get_elem(y_mat, r, c) == INT_MAX) {
                intnn_set_elem(delta_mat, r, c, -1);
                sumDelta -= 1;
            }
        }
    }
    return sumDelta;
}
int intnn_batch_cross_entropy_loss(intnn_mat* loss_mat, const intnn_mat* y_mat, const intnn_mat* y_hat_mat) {
    assert(intnn_dims_equal(y_mat, y_hat_mat));

    intnn_reset_zero(loss_mat, y_mat->mRows, 1);

    int sumLoss = 0;
    for (int r = 0; r < y_mat->mRows; ++r) {
        int rowLoss = 0;
        for (int c = 0; c < y_mat->mCols; ++c) {
            int y = intnn_get_elem(y_mat, r, c);
            int yHat = intnn_get_elem(y_hat_mat, r, c);
            if (y == 1) {
                int delta = yHat - INTNN_MAX;
                rowLoss += (delta * delta) / 2;
            }
        }
        intnn_set_elem(loss_mat, r, 0, rowLoss);
        sumLoss += rowLoss;
    }
    return sumLoss;
}
int intnn_batch_cross_entropy_loss_delta(intnn_mat* delta_mat, const intnn_mat* y_mat, const intnn_mat* y_hat_mat) {
    assert(intnn_dims_equal(y_mat, y_hat_mat));

    if (!intnn_dims_equal(delta_mat, y_hat_mat)) {
        intnn_reset_zero(delta_mat, y_hat_mat->mRows, y_hat_mat->mCols);
    } else {
        intnn_set_all_constant(delta_mat, 0);
    }

    int sumDelta = 0;
    for (int r = 0; r < y_mat->mRows; ++r) {
        for (int c = 0; c < y_mat->mCols; ++c) {
            if (intnn_get_elem(y_mat, r, c) == 1) {
                int delta = intnn_get_elem(y_hat_mat, r, c) - INTNN_MAX;
                intnn_set_elem(delta_mat, r, c, delta);
                sumDelta += delta;
            }
        }
    }
    return sumDelta;
}
