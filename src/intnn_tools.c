#include "intnn_tools.h"
#include "intnn_consts.h"

int intnn_max(int a, int b) {
    return a > b ? a : b;
}

int intnn_min(int a, int b) {
    return a < b ? a : b;
}

int intnn_clamp(int value, int lower, int upper) {
    value = (value < lower) ? lower : value;
    value = (value > upper) ? upper : value;
    return value;
}

int intnn_random_range(int lower, int upper) {
    return (rand() % (upper - lower + 1)) + lower;
}

int intnn_floor_sqrt(int x) {
    if (x <= 0) {
        return 0;
    }

    int ans = 1;
    while (ans * ans <= x) {
        ++ans;
    }
    return ans - 1;
}

int intnn_approx_log(int base, int x, int get_closest) {
    return intnn_int_round_log(base, x, 0, 0, get_closest);
}

int intnn_int_round_log(int base, int x, int x_shift, int y_shift, int get_closest) {
    assert(base > 1);
    int x_minus_p = x - x_shift;

    if (x_minus_p <= 0) {
        return INTNN_MIN;
    } else if (x_minus_p < 1) {
        // In integer-only mode, this is undefined; here, do nothing
    } else if (x_minus_p == 1) {
        return 0;
    } else {
        int exponent = 0;
        int curr = 1;
        int prev = 0;

        while (curr < x_minus_p) {
            prev = curr;
            curr *= base;
            ++exponent;
        }

        int y = exponent - 1;
        if (get_closest) {
            if ((curr - x_minus_p) < (x_minus_p - prev)) {
                y = exponent;
            }
        }

        return y + y_shift;
    }
    return 1;
}

int intnn_round_to_unit(int n, int unit) {
    int a = (n / unit) * unit;
    int b = a + unit;
    return (n - a > b - n) ? b : a;
}
