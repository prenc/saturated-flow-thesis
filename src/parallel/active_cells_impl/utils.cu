#ifndef AC_UTILS
#define AC_UTILS
#include "../utils/memory_management.cuh"

namespace ac_utils {
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
