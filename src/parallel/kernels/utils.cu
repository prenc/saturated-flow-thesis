#ifndef DEVICE_UTILS
#define DEVICE_UTILS
#include "../utils/memory_management.cuh"

namespace device_utils {
    __device__ void calc_global_mem_transition(CA ca, double* headsWrite, int idx_x, int idx_y, int idx_g) {
        double Q{}, diff_head, tmp_t, ht1, ht2;
#ifdef LOOP
        for (int i = 0; i < KERNEL_LOOP_SIZE; i++){
            if (i == KERNEL_LOOP_SIZE - 1)
            {
                if (Q) { Q = 0; }
            }
#endif
        if (idx_x >= 1) {
            diff_head = ca.heads[idx_g - 1] - ca.heads[idx_g];
            tmp_t = ca.K[idx_g] * THICKNESS;
            Q +=
                    diff_head * tmp_t;
        }
        if (idx_y >= 1) {
            diff_head = ca.heads[(idx_y - 1) * COLS + idx_x] - ca.heads[idx_g];
            tmp_t = ca.K[idx_g] * THICKNESS;
            Q +=
                    diff_head * tmp_t;
        }
        if (idx_x + 1 < COLS) {
            diff_head = ca.heads[idx_g + 1] - ca.heads[idx_g];
            tmp_t = ca.K[idx_g] * THICKNESS;
            Q +=
                    diff_head * tmp_t;
        }
        if (idx_y + 1 < ROWS) {
            diff_head = ca.heads[(idx_y + 1) * COLS + idx_x] - ca.heads[idx_g];
            tmp_t = ca.K[idx_g] * THICKNESS;
            Q +=
                    diff_head * tmp_t;
        }
#ifdef LOOP
        }
#endif
        Q -= ca.sources[idx_g];
        ht1 = Q * DELTA_T;
        ht2 = AREA * ca.Sy[idx_g];

        headsWrite[idx_g] = ca.heads[idx_g] + ht1 / ht2;
        if (headsWrite[idx_g] < 0) {
            headsWrite[idx_g] = 0;
        }
    }
}

#endif
