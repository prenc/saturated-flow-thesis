#include "../../common/memory_management.cuh"
#include "../../common/statistics.h"
#include <thrust/device_vector.h>

__global__ void simulation_step_kernel(CA ca, double *headsWrite,
                                       const int *activeCellsIds,
                                       int *activeCellsMask,
                                       int acNumber)
{
    unsigned ac_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned ac_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned ac_idx_g = ac_idx_y * blockDim.x * gridDim.x + ac_idx_x;

    if (ac_idx_g < acNumber)
    {
        double Q{}, diff_head, tmp_t, ht1, ht2;
        unsigned idx_g = activeCellsIds[ac_idx_g];
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
            activeCellsMask[idx_g - 1] = idx_g - 1;
        }
        if (idx_y >= 1)
        {
            diff_head = ca.heads[(idx_y - 1) * COLS + idx_x] - ca.heads[idx_g];
            tmp_t = ca.K[idx_g] * THICKNESS;
            Q += diff_head * tmp_t;
            activeCellsMask[(idx_y - 1) * COLS + idx_x] = (idx_y - 1) * COLS + idx_x;
        }
        if (idx_x + 1 < COLS)
        {
            diff_head = ca.heads[idx_g + 1] - ca.heads[idx_g];
            tmp_t = ca.K[idx_g] * THICKNESS;
            Q += diff_head * tmp_t;
            activeCellsMask[idx_g + 1] = idx_g + 1;
        }
        if (idx_y + 1 < ROWS)
        {
            diff_head = ca.heads[(idx_y + 1) * COLS + idx_x] - ca.heads[idx_g];
            tmp_t = ca.K[idx_g] * THICKNESS;
            Q += diff_head * tmp_t;
            activeCellsMask[(idx_y + 1) * COLS + idx_x] = (idx_y + 1) * COLS + idx_x;
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

template<typename T>
struct is_not_minus_one
{
    __host__ __device__
    auto operator()(T x) const -> bool
    {
        return x != -1;
    }
};

int main(int argc, char *argv[])
{
    auto h_ca = new CA();
    double *headsWrite;
    allocateManagedMemory(h_ca, headsWrite);

    thrust::device_vector<int> activeCellsMask(ROWS * COLS, -1);
    thrust::device_vector<int> activeCellsIds(ROWS * COLS, -1);

    initializeCA(h_ca);
    memcpy(headsWrite, h_ca->heads, sizeof(double) * ROWS * COLS);
    for (size_t i{0}; i < ROWS * COLS; ++i)
    {
        if (h_ca->sources[i] != 0)
        {
            activeCellsMask[i] = i;
        }
    }

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    const int blockCount = ceil((double) (ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE));
    int gridSize = ceil(sqrt(blockCount));
    dim3 gridDims(gridSize, gridSize);

    std::vector<StatPoint> stats;
    Timer stepTimer, activeCellsEvalTimer, transitionTimer;
    stepTimer.start();

    bool isWholeGridActive = false;
    dim3 *simulationGridDims;
    int devActiveCellsCount;
    for (int i{}; i < SIMULATION_ITERATIONS; ++i)
    {
        if (!isWholeGridActive)
        {
            activeCellsEvalTimer.start();
            thrust::copy_if(thrust::device, activeCellsMask.begin(), activeCellsMask.end(),
                            activeCellsIds.begin(), is_not_minus_one<int>());
            devActiveCellsCount = thrust::count_if(activeCellsIds.begin(), activeCellsIds.end(),
                                                   is_not_minus_one<int>());
            activeCellsEvalTimer.stop();

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

        transitionTimer.start();
        simulation_step_kernel <<< *simulationGridDims, blockSize >>>(
                *h_ca, headsWrite, thrust::raw_pointer_cast(&activeCellsIds[0]),
                thrust::raw_pointer_cast(&activeCellsMask[0]),
                devActiveCellsCount);
        ERROR_CHECK(cudaDeviceSynchronize());
        transitionTimer.stop();

        double *tmpHeads = h_ca->heads;
        h_ca->heads = headsWrite;
        headsWrite = tmpHeads;

        if (i % STATISTICS_WRITE_FREQ == 0)
        {
            stepTimer.stop();
            auto stat = new StatPoint(
                    devActiveCellsCount / (double) (ROWS * COLS),
                    stepTimer.elapsedNanoseconds(),
                    transitionTimer.elapsedNanoseconds(),
                    activeCellsEvalTimer.elapsedNanoseconds());
            stats.push_back(*stat);
            stepTimer.start();
        }
    }

    if (WRITE_OUTPUT_TO_FILE)
    {
        saveHeadsInFile(h_ca->heads, argv[0]);
    }

    if (WRITE_STATISTICS_TO_FILE)
    {
        writeStatisticsToFile(stats, argv[0]);
    }

    freeAllocatedMemory(h_ca, headsWrite);
}