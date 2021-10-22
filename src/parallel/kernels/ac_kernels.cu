#ifndef AC_KERNELS
#define AC_KERNELS

#include "./utils.cu"
#include "../common/memory_management.cuh"

template<typename T>
struct is_not_minus_one
{
    __host__ __device__
    auto operator()(T x) const -> bool
    {
        return x != -1;
    }
};

namespace ac_kernels
{
    __global__ void sc(CA ca, double *headsWrite,
                                           const int *activeCellsIds,
                                           int *activeCellsMask,
                                           int acNumber)
    {
        unsigned ac_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned ac_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned ac_idx_g = ac_idx_y * blockDim.x * gridDim.x + ac_idx_x;

        if (ac_idx_g < acNumber)
        {
            int idx_g = activeCellsIds[ac_idx_g];
            int idx_x = idx_g % COLS;
            int idx_y = idx_g / COLS;
            device_utils::calc_global_mem_tranistion(ca, headsWrite, idx_x, idx_y, idx_g);
            device_utils::mark_cell_neighbours_as_active(headsWrite, activeCellsMask, idx_x, idx_y, idx_g);
        }
    }

    __global__ void naive(CA ca, double *headsWrite)
    {
        unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned idx_g = idx_y * COLS + idx_x;

        if (idx_x < ROWS && idx_y < COLS)
        {
            if (!device_utils::isActiveCell(ca, idx_x, idx_y, idx_g))
            {
                return;
            }
            device_utils::calc_global_mem_tranistion(ca, headsWrite, idx_x, idx_y, idx_g);
        }
    }

}

#endif
