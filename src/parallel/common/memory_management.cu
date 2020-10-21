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

void copyDataFromCpuToGpu(CA *&h_ca, CA *&d_ca)
{
    ERROR_CHECK(cudaMemcpy(d_ca->heads, h_ca->heads, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_ca->K, h_ca->K, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_ca->Sy, h_ca->Sy, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_ca->sources, h_ca->sources, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
}

CA *initializeCA()
{
    int wellsRows[] = {WELLS_Y};
    int wellsCols[] = {WELLS_X};
    double wellsQW[] = {WELLS_QW};

    auto ca = new CA();
    ca->heads = new double[ROWS * COLS]();
    ca->Sy = new double[ROWS * COLS]();
    ca->K = new double[ROWS * COLS]();
    ca->sources = new double[ROWS * COLS]();

    for (int i{}; i < ROWS; ++i)
        for (int j{}; j < COLS; ++j)
        {
            ca->heads[i * ROWS + j] = headFixed;
            if (j == COLS - 1)
            {
                ca->heads[i * ROWS + j] = headCalculated;
            }
            ca->Sy[i * ROWS + j] = Syinitial;
            ca->K[i * ROWS + j] = Kinitial;
            ca->sources[i * ROWS + j] = 0;
        }

    int x, y;
    double source;
    for (int i{}; i < NUMBER_OF_WELLS; ++i)
    {
        x = wellsRows[i];
        y = wellsCols[i];
        source = wellsQW[i];
        ca->sources[y * ROWS + x] = source;
    }
    return ca;
}

void copyDataFromGpuToCpu(CA *&h_ca, CA *&d_ca)
{
    ERROR_CHECK(cudaMemcpy(h_ca->heads, d_ca->heads, sizeof(double) * ROWS * COLS, cudaMemcpyDeviceToHost));
}

void free_allocated_memory(CA *&d_ca, double *&headsWrite)
{
    cudaFree(headsWrite);
    cudaFree(d_ca->heads);
    cudaFree(d_ca->Sy);
    cudaFree(d_ca->K);
    cudaFree(d_ca->sources);
}
