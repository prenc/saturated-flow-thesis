#include "../../../common/memory_management.cuh"
#include "../../../common/statistics.h"
#include "../../../kernels/transition_kernels.cu"

__managed__ int devActiveCellsCount = 0;

__device__ unsigned activeCellsIdx[ROWS * COLS];

__device__ void my_push_back(unsigned cellIdx)
{
    int insert_ptr = atomicAdd(&devActiveCellsCount, 1);
    activeCellsIdx[insert_ptr] = cellIdx;
}

__global__ void simulation_step_kernel(struct CA ca, double *headsWrite)
{
    unsigned ac_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned ac_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned ac_idx_g = ac_idx_y * blockDim.x * gridDim.x + ac_idx_x;

    if (ac_idx_g < devActiveCellsCount)
    {
        double Q{}, diff_head, tmp_t, ht1, ht2;
        unsigned idx_g = activeCellsIdx[ac_idx_g];
        unsigned idx_x = idx_g % COLS;
        unsigned idx_y = idx_g / COLS;
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

__global__ void findActiveCells(struct CA d_ca)
{
    unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx_g = idx_y * COLS + idx_x;

    if (idx_x < ROWS && idx_y < COLS)
    {
        if (d_ca.heads[idx_g] < INITIAL_HEAD || d_ca.sources[idx_g] != 0)
        {
            my_push_back(idx_g);
            return;
        }
        if (idx_x > 0)
        {
            if (d_ca.heads[idx_g - 1] < INITIAL_HEAD)
            {
                my_push_back(idx_g);
                return;
            }
        }
        if (idx_y > 0)
        {
            if (d_ca.heads[idx_g - COLS] < INITIAL_HEAD)
            {
                my_push_back(idx_g);
                return;
            }
        }
        if (idx_x < COLS - 1)
        {
            if (d_ca.heads[idx_g + 1] < INITIAL_HEAD)
            {
                my_push_back(idx_g);
                return;
            }
        }
        if (idx_y < ROWS - 1)
        {
            if (d_ca.heads[idx_g + COLS] < INITIAL_HEAD)
            {
                my_push_back(idx_g);
                return;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    CA *d_ca = new CA();
    CA *h_ca = new CA();
    double *headsWrite;

    h_ca->heads = new double[ROWS * COLS]();
    h_ca->Sy = new double[ROWS * COLS]();
    h_ca->K = new double[ROWS * COLS]();
    h_ca->sources = new double[ROWS * COLS]();

    initializeCA(h_ca);

    allocateMemory(d_ca, headsWrite);
    copyDataFromCpuToGpu(h_ca, d_ca, headsWrite);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    const int blockCount = ceil((double) (ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE));
    int gridSize = ceil(sqrt(blockCount));
    dim3 gridDims(gridSize, gridSize);

    std::vector<StatPoint> stats;
    Timer stepTimer;
    stepTimer.start();

    bool isWholeGridActive = false;
    dim3 *simulationGridDims;
    for (size_t i{}; i < SIMULATION_ITERATIONS; ++i)
    {
        if (!isWholeGridActive)
        {
            devActiveCellsCount = 0;
            findActiveCells <<< gridDims, blockSize >>>(*d_ca);
            ERROR_CHECK(cudaDeviceSynchronize());

            isWholeGridActive = devActiveCellsCount == ROWS * COLS;

            int activeBlockCount = ceil((double) devActiveCellsCount / (BLOCK_SIZE * BLOCK_SIZE));
            int activeGridSize = ceil(sqrt(activeBlockCount));
            dim3 activeGridDim(activeGridSize, activeGridSize);

            simulationGridDims = &activeGridDim;
        }
        else
        {
            simulationGridDims = &gridDims;
        }

        simulation_step_kernel <<< *simulationGridDims, blockSize >>>(*d_ca, headsWrite);
        ERROR_CHECK(cudaDeviceSynchronize());

        double *tmpHeads = d_ca->heads;
        d_ca->heads = headsWrite;
        headsWrite = tmpHeads;

        if (i % STATISTICS_WRITE_FREQ == STATISTICS_WRITE_FREQ - 1)
        {
            stepTimer.stop();
            auto stat = new StatPoint(
                    devActiveCellsCount / (double) (ROWS * COLS),
                    stepTimer.elapsedNanoseconds());
            stats.push_back(*stat);
            stepTimer.start();
        }
    }

    if (WRITE_OUTPUT_TO_FILE)
    {
        copyDataFromGpuToCpu(h_ca, d_ca);
        saveHeadsInFile(h_ca->heads, argv[0]);
    }

    if (WRITE_STATISTICS_TO_FILE)
    {
        writeStatisticsToFile(stats, argv[0]);
    }

    freeAllocatedMemory(d_ca, headsWrite);
}