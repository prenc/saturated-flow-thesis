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
}

#endif
