#ifndef INTNN_EXAMPLES_H
#define INTNN_EXAMPLES_H

#include <stdio.h>
#include <stdbool.h>
#include "intnn_mat.h"
#include "intnn_actv.h"
#include "intnn_tools.h"
#include "intnn_consts.h"
#include "intnn_loss.h"
#include "intnn_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

int example_intnn_fc_dfa_mnist();

#ifdef __cplusplus
}
#endif

#endif // INTNN_EXAMPLES
