#ifndef DEVICE_UTILS
#define DEVICE_UTILS
#include "../common/memory_management.cuh"

namespace device_utils {
    __device__ bool isActiveCell(CA ca, int idx_x, int idx_y, int idx_g) {
        if (ca.sources[idx_g] != 0 || ca.heads[idx_g] < INITIAL_HEAD ) {
            return true;
        }
        if (idx_x > 0 && ca.heads[idx_g - 1] < INITIAL_HEAD) {
            return true;
        }
        if (idx_y > 0 && ca.heads[idx_g - COLS] < INITIAL_HEAD) {
            return true;
        }
        if (idx_x<COLS - 1 && ca.heads[idx_g + 1] < INITIAL_HEAD) {
            return true;
        }
        if (idx_y<ROWS - 1 && ca.heads[idx_g + COLS] < INITIAL_HEAD) {
            return true;
        }
        return false;
    }

    __device__ void calc_global_mem_tranistion(CA ca, double* headsWrite, int idx_x, int idx_y, int idx_g) {
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
                diff_head *tmp_t;
            }
            if (idx_y >= 1) {
                diff_head = ca.heads[(idx_y - 1) * COLS + idx_x] - ca.heads[idx_g];
                tmp_t = ca.K[idx_g] * THICKNESS;
                Q +=
                diff_head *tmp_t;
            }
            if (idx_x + 1 < COLS) {
                diff_head = ca.heads[idx_g + 1] - ca.heads[idx_g];
                tmp_t = ca.K[idx_g] * THICKNESS;
                Q +=
                diff_head *tmp_t;
            }
            if (idx_y + 1 < ROWS){
                diff_head = ca.heads[(idx_y + 1) * COLS + idx_x] - ca.heads[idx_g];
                tmp_t = ca.K[idx_g] * THICKNESS;
                Q +=
                diff_head *tmp_t;
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

    __device__ void mark_cell_neighbours_as_active(double  *headsWrite, int *activeCellsMask, int idx_x, int idx_y, int idx_g) {
        if (headsWrite[idx_g] < INITIAL_HEAD) {
            if (idx_x >= 1) {
                activeCellsMask[idx_g - 1] = idx_g - 1;
            }
            if (idx_y >= 1) {
                activeCellsMask[(idx_y - 1) * COLS + idx_x] = (idx_y - 1) * COLS + idx_x;
            }
            if (idx_x + 1 < COLS) {
                activeCellsMask[idx_g + 1] = idx_g + 1;
            }
            if (idx_y + 1 < ROWS) {
                activeCellsMask[(idx_y + 1) * COLS + idx_x] = (idx_y + 1) * COLS + idx_x;
            }
        }
    }

}

#endif
