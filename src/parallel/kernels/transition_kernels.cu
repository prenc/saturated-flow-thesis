#ifndef SATURATED_FLOW_ITERATION_STEP_CU
#define SATURATED_FLOW_ITERATION_STEP_CU

#include "../common/memory_management.cuh"
#include "./utils.cu"

namespace kernels
{
    __global__ void standard_step(struct CA ca, double *headsWrite)
    {
        unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned idx_g = idx_y * COLS + idx_x;

        if (idx_x < COLS && idx_y < ROWS) {
            device_utils::calc_global_mem_tranistion(ca, headsWrite, idx_x, idx_y, idx_g);
        }
    }

    __global__ void hybrid_step(struct CA ca, double *headsWrite)
    {
        __shared__ double s_heads[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double s_K[BLOCK_SIZE][BLOCK_SIZE];
        unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned idx_g = idx_y * COLS + idx_x;

        if (idx_x < COLS && idx_y < ROWS)
        {
            double Q{}, diff_head, tmp_t, ht1, ht2;
            s_heads[threadIdx.y][threadIdx.x] = ca.heads[idx_g];
            s_K[threadIdx.y][threadIdx.x] = ca.K[idx_g];
            __syncthreads();

#ifdef LOOP
            for (int i = 0; i < KERNEL_LOOP_SIZE; i++)
            {
                if (i == KERNEL_LOOP_SIZE - 1)
                {
                    if (Q) { Q = 0; }
                }
#endif
                if (idx_x >= 1)
                { // left neighbor
                    if (threadIdx.x >= 1)
                        diff_head = s_heads[threadIdx.y][threadIdx.x - 1] - s_heads[threadIdx.y][threadIdx.x];
                    else
                        diff_head = ca.heads[idx_g - 1] - s_heads[threadIdx.y][threadIdx.x];
                    tmp_t = s_K[threadIdx.y][threadIdx.x] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
                if (idx_y >= 1)
                { // upper neighbor
                    if (threadIdx.y >= 1)
                        diff_head = s_heads[threadIdx.y - 1][threadIdx.x] - s_heads[threadIdx.y][threadIdx.x];
                    else
                        diff_head = ca.heads[(idx_y - 1) * COLS + idx_x] - s_heads[threadIdx.y][threadIdx.x];
                    tmp_t = s_K[threadIdx.y][threadIdx.x] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
                if (idx_x + 1 < COLS)
                { // right neighbor
                    if (threadIdx.x < BLOCK_SIZE - 1)
                        diff_head = s_heads[threadIdx.y][threadIdx.x + 1] - s_heads[threadIdx.y][threadIdx.x];
                    else
                        diff_head = ca.heads[idx_g + 1] - s_heads[threadIdx.y][threadIdx.x];
                    tmp_t = s_K[threadIdx.y][threadIdx.x] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
                if (idx_y + 1 < ROWS)
                { // bottom neighbor
                    if (threadIdx.y < BLOCK_SIZE - 1)
                        diff_head = s_heads[threadIdx.y + 1][threadIdx.x] - s_heads[threadIdx.y][threadIdx.x];
                    else
                        diff_head = ca.heads[(idx_y + 1) * COLS + idx_x] - s_heads[threadIdx.y][threadIdx.x];
                    tmp_t = s_K[threadIdx.y][threadIdx.x] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
#ifdef LOOP
            }
#endif
            Q -= ca.sources[idx_g];

            ht1 = Q * DELTA_T;
            ht2 = AREA * ca.Sy[idx_g];

            headsWrite[idx_g] = s_heads[threadIdx.y][threadIdx.x] + ht1 / ht2;
            if (headsWrite[idx_g] < 0)
            { headsWrite[idx_g] = 0; }
        }
    }

    __global__ void shared_step(struct CA ca, double *headsWrite)
    {
        __shared__ double s_heads[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
        __shared__ double s_K[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
        unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned idx_g = idx_y * COLS + idx_x;

        if (idx_x < COLS && idx_y < ROWS)
        {
            double Q{}, diff_head, tmp_t, ht1, ht2;
            unsigned x = threadIdx.x + 1;
            unsigned y = threadIdx.y + 1;

            s_heads[y][x] = ca.heads[idx_g];
            s_K[y][x] = ca.K[idx_g];

            if (threadIdx.x == 0 && blockIdx.x != 0) // left
                s_heads[y][x - 1] = ca.heads[idx_g - 1];
            if (threadIdx.x == BLOCK_SIZE - 1 && blockIdx.x != gridDim.x - 1) // right
                s_heads[y][x + 1] = ca.heads[idx_g + 1];
            if (threadIdx.y == 0 && blockIdx.y != 0) // upper
                s_heads[y - 1][x] = ca.heads[idx_g - COLS];
            if (threadIdx.y == BLOCK_SIZE - 1 && blockIdx.y != gridDim.y - 1) // bottom
                s_heads[y + 1][x] = ca.heads[idx_g + COLS];
            __syncthreads();

#ifdef LOOP
            for (int i = 0; i < KERNEL_LOOP_SIZE; i++)
            {
                if (i == KERNEL_LOOP_SIZE - 1)
                {
                    if (Q) { Q = 0; }
                }
#endif
                if (idx_x >= 1)
                { // left neighbor
                    diff_head = s_heads[y][x - 1] - s_heads[y][x];
                    tmp_t = s_K[y][x] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
                if (idx_y >= 1)
                { // upper neighbor
                    diff_head = s_heads[y - 1][x] - s_heads[y][x];
                    tmp_t = s_K[y][x] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
                if (idx_x + 1 < COLS)
                { // right neighbor
                    diff_head = s_heads[y][x + 1] - s_heads[y][x];
                    tmp_t = s_K[y][x] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
                if (idx_y + 1 < ROWS)
                { // bottom neighbor
                    diff_head = s_heads[y + 1][x] - s_heads[y][x];
                    tmp_t = s_K[y][x] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
#ifdef LOOP
            }
#endif
            Q -= ca.sources[idx_g];
            ht1 = Q * DELTA_T;
            ht2 = AREA * ca.Sy[idx_g];

            headsWrite[idx_g] = s_heads[y][x] + ht1 / ht2;
            if (headsWrite[idx_g] < 0)
            { headsWrite[idx_g] = 0; }
        }
    }
}


#endif //SATURATED_FLOW_ITERATION_STEP_CU
