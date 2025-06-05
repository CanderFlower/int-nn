#ifndef INTNN_TOOLS_H
#define INTNN_TOOLS_H

#include <stdlib.h>  // for rand
#include <assert.h>  // for assert
#include "intnn_mat.h"

#ifdef __cplusplus
extern "C" {
#endif

int intnn_max(int a, int b);
int intnn_min(int a, int b);
int intnn_clamp(int value, int lower, int upper);
int intnn_random_range(int lower, int upper);
int intnn_floor_sqrt(int x);
int intnn_int_round_log(int base, int x, int x_shift, int y_shift, int get_closest);
int intnn_approx_log(int base, int x, int get_closest);
int intnn_round_to_unit(int n, int unit);
void intnn_tools_shuffle_indices(int* indices, int size);

#ifdef __cplusplus
}
#endif

#endif // INTNN_TOOLS_H
