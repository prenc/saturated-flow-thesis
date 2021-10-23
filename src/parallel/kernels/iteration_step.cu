#ifndef SATURATED_FLOW_ITERATION_STEP_CU
#define SATURATED_FLOW_ITERATION_STEP_CU

#include "../common/memory_management.cuh"

namespace kernels
{
    __global__ void standard_step(struct CA ca, double *headsWrite)
    {
        unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned idx_g = idx_y * COLS + idx_x;

        if (idx_x < COLS && idx_y < ROWS)
        {
            double Q{}, diff_head, tmp_t, ht1, ht2;
#ifdef LOOP
            for (int i = 0; i < KERNEL_LOOP_SIZE; i++)
            {
                if (i == KERNEL_LOOP_SIZE - 1)
                {
                    if (Q) { Q = 0; }
                }
#endif
                if (idx_x >= 1)
                {
                    diff_head = ca.heads[idx_g - 1] - ca.heads[idx_g];
                    tmp_t = ca.K[idx_g] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
                if (idx_y >= 1)
                {
                    diff_head = ca.heads[(idx_y - 1) * COLS + idx_x] - ca.heads[idx_g];
                    tmp_t = ca.K[idx_g] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
                if (idx_x + 1 < COLS)
                {
                    diff_head = ca.heads[idx_g + 1] - ca.heads[idx_g];
                    tmp_t = ca.K[idx_g] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
                if (idx_y + 1 < ROWS)
                {
                    diff_head = ca.heads[(idx_y + 1) * COLS + idx_x] - ca.heads[idx_g];
                    tmp_t = ca.K[idx_g] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
#ifdef LOOP
            }
#endif
            Q -= ca.sources[idx_g];
            ht1 = Q * DELTA_T;
            ht2 = AREA * ca.Sy[idx_g];

            headsWrite[idx_g] = ca.heads[idx_g] + ht1 / ht2;
            if (headsWrite[idx_g] < 0)
            { headsWrite[idx_g] = 0; }
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


    __global__ void dummy_active(CA ca,
                                       double *headsWrite,
                                       const int *activeCellsIds,
                                       int *activeCellsMask,
                                       int acNumber)
    {
        unsigned ac_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned ac_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned ac_idx_g = ac_idx_y * blockDim.x * gridDim.x + ac_idx_x;

        if (ac_idx_g < acNumber)
        {
            double a, b, c = 734, k, d;
            double n1 = 1.982, e;
            double f, g, h, j;
            double l, z, m, n,u ;
            int n2;
            double o, q;
            double aa = 0.0334;
            double bb = 0.368;
            double cc = 0.102;
            double dd = 0.009154;

            for (int qaz = 0; qaz < LOOP_DUMMY_COMPUTATION; qaz++)
            {
                g = pow(aa * (-c), (1 - n1));
                h = pow(aa * (-c), n1);
                j = pow((1 / (1 + h)), (1 / n1 - 2));
                d = (g / (aa * (n1 - 1) * (bb - cc))) * j;

                f = pow(aa * (-c), n);
                a = cc + ((bb - cc) * pow((1 / (1 + f)), (1 - 1 / n1)));
                q = pow(aa * (734), n1);
                o = cc + ((bb - cc) * pow((1 / (1 + q)), (1 - 1 / n1)));
                e = a / bb;
                u = e - o / bb;

                b = (a - cc) / (bb - cc);
                l = n1 / (n1 - 1);
                m = pow(b, l);
                z = 1 - (1 / n1);
                n2 = pow((1 - m), z);
                k = dd * pow(b, 0.5) * pow((1 - n), 2);
            }
        }

    }
    __global__ void dummy_all(struct CA ca, double *headsWrite)
    {
        unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned idx_g = idx_y * COLS + idx_x;

        if (idx_x < COLS && idx_y < ROWS)
        {

            double a, b, c = 734, k, d;
            double n1 = 1.982, e;
            double f, g, h, j;
            double l, z, m, n,u ;
            int n2;
            double o, q;
            double aa = 0.0334;
            double bb = 0.368;
            double cc = 0.102;
            double dd = 0.009154;

            for (int qaz = 0; qaz < LOOP_DUMMY_COMPUTATION; qaz++)
            {
                g = pow(aa * (-c), (1 - n1));
                h = pow(aa * (-c), n1);
                j = pow((1 / (1 + h)), (1 / n1 - 2));
                d = (g / (aa * (n1 - 1) * (bb - cc))) * j;

                f = pow(aa * (-c), n);
                a = cc + ((bb - cc) * pow((1 / (1 + f)), (1 - 1 / n1)));
                q = pow(aa * (734), n1);
                o = cc + ((bb - cc) * pow((1 / (1 + q)), (1 - 1 / n1)));
                e = a / bb;
                u = e - o / bb;

                b = (a - cc) / (bb - cc);
                l = n1 / (n1 - 1);
                m = pow(b, l);
                z = 1 - (1 / n1);
                n2 = pow((1 - m), z);
                k = dd * pow(b, 0.5) * pow((1 - n), 2);
            }
        }

    }
}


#endif //SATURATED_FLOW_ITERATION_STEP_CU
