#ifndef INTNN_CONSTS_H
#define INTNN_CONSTS_H

#include <limits.h>  // for SCHAR_MIN, SCHAR_MAX

#ifdef __cplusplus
extern "C" {
#endif

#define INTNN_K_BIT 8
#define INTNN_MIN (SCHAR_MIN + 1) // -127
#define INTNN_MAX (SCHAR_MAX)     // 127
#define INTNN_UNSIGNED_4BIT_MAX 15

extern const char* INTNN_TYPE_FC;
extern const char* INTNN_TYPE_CONV;

#ifdef __cplusplus
}
#endif

#endif // INTNN_CONSTS_H
