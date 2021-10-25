#ifndef DUMMY_KERNELS
#define DUMMY_KERNELS
#include "../active_cells_impl/utils.cu"
#include "../utils/memory_management.cuh"

namespace dummy_kernels
{
    __device__ int dummy_computations(double active_cell_value){
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
            n2 = pow((1 - m), z) + dd;
            k = pow(active_cell_value + d + u + n2, 2);
        }

        return (int)k % 10;
    }

    __global__ void dummy_active_sc(CA ca,
                                    double *headsWrite,
                                    const int *activeCellsIds,
                                    int acNumber)
    {
        unsigned ac_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned ac_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned ac_idx_g = ac_idx_y * blockDim.x * gridDim.x + ac_idx_x;

        if (ac_idx_g < acNumber)
        {
            int idx_g = activeCellsIds[ac_idx_g];
            int dummy = dummy_computations(ca.heads[idx_g]);
            headsWrite[idx_g] += dummy;
        }
    }

    __global__ void dummy_active_naive(struct CA ca, double *headsWrite)
    {
        unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned idx_g = idx_y * COLS + idx_x;

        if (idx_x < COLS && idx_y < ROWS) {
            if(ac_utils::isActiveCell(ca, idx_x, idx_y, idx_g)){
                int dummy = dummy_computations(ca.heads[idx_g]);
                headsWrite[idx_g] += dummy;
            }
        }
    }

    __global__ void dummy_all(struct CA ca, double *headsWrite)
    {
        unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned idx_g = idx_y * COLS + idx_x;

        if (idx_x < COLS && idx_y < ROWS) {
            int dummy = dummy_computations(ca.heads[idx_g]);
            headsWrite[idx_g] += dummy;
        }
    }


}

#endif
