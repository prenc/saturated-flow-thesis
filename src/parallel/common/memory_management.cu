#include "memory_management.cuh"

void allocateManagedMemory(CA *&ca, double *&heads_write)
{
    ERROR_CHECK(cudaMallocManaged((void **) &heads_write, sizeof(double) * ROWS * COLS));
    ERROR_CHECK(cudaMallocManaged((void **) &ca->heads, sizeof(double) * ROWS * COLS));
    ERROR_CHECK(cudaMallocManaged((void **) &ca->K, sizeof(double) * ROWS * COLS));
    ERROR_CHECK(cudaMallocManaged((void **) &ca->Sy, sizeof(double) * ROWS * COLS));
    ERROR_CHECK(cudaMallocManaged((void **) &ca->sources, sizeof(double) * ROWS * COLS));
}

void allocateMemory(CA *&ca, double *&headsWrite)
{
    ERROR_CHECK(cudaMalloc((void **) &headsWrite, sizeof(double) * ROWS * COLS));
    ERROR_CHECK(cudaMalloc((void **) &ca->heads, sizeof(double) * ROWS * COLS));
    ERROR_CHECK(cudaMalloc((void **) &ca->K, sizeof(double) * ROWS * COLS));
    ERROR_CHECK(cudaMalloc((void **) &ca->Sy, sizeof(double) * ROWS * COLS));
    ERROR_CHECK(cudaMalloc((void **) &ca->sources, sizeof(double) * ROWS * COLS));
}

void copyDataFromCpuToGpu(CA *&h_ca, CA *&d_ca, double *headsWrite)
{
    ERROR_CHECK(cudaMemcpy(headsWrite, h_ca->heads, sizeof(double) * ROWS * COLS,
                           cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_ca->heads, h_ca->heads, sizeof(double) * ROWS * COLS,
                           cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_ca->K, h_ca->K, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
    ERROR_CHECK(
            cudaMemcpy(d_ca->Sy, h_ca->Sy, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_ca->sources, h_ca->sources, sizeof(double) * ROWS * COLS,
                           cudaMemcpyHostToDevice));
}

void initializeCA(CA *&ca)
{
    int wellsRows[] = {WELLS_Y};
    int wellsCols[] = {WELLS_X};
    double wellsQW[] = {WELLS_QW};

    for (int i{}; i < ROWS; ++i)
    {
        for (int j{}; j < COLS; ++j)
        {
            ca->heads[i * ROWS + j] = INITIAL_HEAD;
            ca->Sy[i * ROWS + j] = INITIAL_SY;
            ca->K[i * ROWS + j] = INITIAL_K;
            ca->sources[i * ROWS + j] = 0;
        }
    }

    for (int i{}; i < NUMBER_OF_WELLS; ++i)
    {
        int x = wellsRows[i];
        int y = wellsCols[i];
        ca->sources[y * ROWS + x] = wellsQW[i];
    }
}

void copyDataFromGpuToCpu(CA *&h_ca, CA *&d_ca)
{
    ERROR_CHECK(cudaMemcpy(h_ca->heads, d_ca->heads, sizeof(double) * ROWS * COLS,
                           cudaMemcpyDeviceToHost));
}

void freeAllocatedMemory(CA *&d_ca, double *&headsWrite)
{
    cudaFree(headsWrite);
    cudaFree(d_ca->heads);
    cudaFree(d_ca->Sy);
    cudaFree(d_ca->K);
    cudaFree(d_ca->sources);
}

dim3 calculate_grid_dim()
{
    return calculate_grid_dim(ROWS * COLS);
}

dim3 calculate_grid_dim(int cell_count)
{
    const int blockCount = ceil((double) (cell_count) / (BLOCK_SIZE * BLOCK_SIZE));
    int gridSize = ceil(sqrt(blockCount));
    return dim3(gridSize, gridSize);
}

void save_output_and_free_memory(char *argv[], struct CA *h_ca, struct CA *d_ca, double *headsWrite, std::vector<StatPoint> &stats){
    if (WRITE_OUTPUT_TO_FILE)
    {
        copyDataFromGpuToCpu(h_ca, d_ca);
    }
    save_output_and_free_memory(argv, h_ca, headsWrite, stats);
}

void save_output_and_free_memory(char *argv[], struct CA *h_ca, double *headsWrite, std::vector<StatPoint> &stats){
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
