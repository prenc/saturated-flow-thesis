#include "params.h"

struct CA {
    double *head;
    double *Sy;
    double *K;
    double *Source;
} h_ca;

double *d_write_head;
CA *d_read_ca;

void init_host_ca();

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__global__ void simulation_step_kernel(struct CA *d_ca, double *d_write_head, int grid_size) {
    __shared__ double s_heads[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    __shared__ double s_K[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx_g = idx_y * COLS + idx_x;

    unsigned x = threadIdx.x + 1;
    unsigned y = threadIdx.y + 1;

    s_heads[y][x] = d_ca->head[idx_g];
    s_K[y][x] = d_ca->K[idx_g];

    if (threadIdx.x == 0 && blockIdx.x != 0) // left
        s_heads[y][x - 1] = d_ca->head[idx_g - 1];
    if (threadIdx.x == BLOCK_SIZE - 1 && blockIdx.x != grid_size - 1) // right
        s_heads[y][x + 1] = d_ca->head[idx_g + 1];
    if (threadIdx.y == 0 && blockIdx.y != 0) // upper
        s_heads[y - 1][x] = d_ca->head[idx_g - COLS];
    if (threadIdx.y == BLOCK_SIZE - 1 && blockIdx.y != grid_size - 1) // bottom
        s_heads[y + 1][x] = d_ca->head[idx_g + COLS];

    __syncthreads();

    double Q = 0;
    double diff_head;
    double tmp_t;

    if (idx_x < COLS && idx_y < ROWS)
        if (idx_y != 0 && idx_y != ROWS - 1) {
            if (idx_x >= 1) { // left neighbor
                diff_head = s_heads[y][x - 1] - s_heads[y][x];
                tmp_t = s_K[y][x] * THICKNESS;
                Q += diff_head * tmp_t;
            }
            if (idx_y >= 1) { // upper neighbor
                diff_head = s_heads[y - 1][x] - s_heads[y][x];
                tmp_t = s_K[y][x] * THICKNESS;
                Q += diff_head * tmp_t;
            }
            if (idx_x + 1 < COLS) { // right neighbor
                diff_head = s_heads[y][x + 1] - s_heads[y][x];
                tmp_t = s_K[y][x] * THICKNESS;
                Q += diff_head * tmp_t;
            }
            if (idx_y + 1 < ROWS) { // bottom neighbor
                diff_head = s_heads[y + 1][x] - s_heads[y][x];
                tmp_t = s_K[y][x] * THICKNESS;
                Q += diff_head * tmp_t;
            }

            Q -= d_ca->Source[idx_g];

            double ht1 = Q * DELTA_T;
            double ht2 = AREA * d_ca->Sy[idx_g];

            d_write_head[idx_g] = s_heads[y][x] + ht1 / ht2;
        }
}

void copy_data_from_CPU_to_GPU() {
    double *d_head, *d_Sy, *d_K, *d_Source;

    CUDA_CHECK_RETURN(cudaMalloc(&d_read_ca, sizeof(*d_read_ca)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_head, sizeof(*d_head) * ROWS * COLS));
    CUDA_CHECK_RETURN(cudaMalloc(&d_write_head, sizeof(double) * ROWS * COLS));
    CUDA_CHECK_RETURN(cudaMalloc(&d_Sy, sizeof(double) * ROWS * COLS));
    CUDA_CHECK_RETURN(cudaMalloc(&d_K, sizeof(double) * ROWS * COLS));
    CUDA_CHECK_RETURN(cudaMalloc(&d_Source, sizeof(double) * ROWS * COLS));

    CUDA_CHECK_RETURN(cudaMemcpy(d_head, h_ca.head, sizeof(*d_head) * ROWS * COLS, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&(d_read_ca->head), &d_head, sizeof(d_read_ca->head), cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(d_Sy, h_ca.Sy, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&(d_read_ca->Sy), &d_Sy, sizeof(d_read_ca->Sy), cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(d_K, h_ca.K, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&(d_read_ca->K), &d_K, sizeof(d_read_ca->K), cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(d_Source, h_ca.Source, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(
            cudaMemcpy(&(d_read_ca->Source), &d_Source, sizeof(d_read_ca->Source), cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(d_write_head, h_ca.head, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
}

void copy_data_from_GPU_to_CPU() {
    CUDA_CHECK_RETURN(cudaMemcpy(h_ca.head, d_write_head, sizeof(double) * ROWS * COLS, cudaMemcpyDeviceToHost));
}

void perform_simulation_on_GPU() {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    const int blockCount = ceil((ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE));
    double gridSize = ceil(sqrt(blockCount));
    dim3 gridDim(gridSize, gridSize);

    for (int i = 0; i < SIMULATION_ITERATIONS; i++) {
        simulation_step_kernel << < gridDim, blockDim >> > (d_read_ca, d_write_head, gridSize);

        cudaDeviceSynchronize();

        double *tmp1 = d_write_head;
        CUDA_CHECK_RETURN(
                cudaMemcpy(&d_write_head, &(d_read_ca->head), sizeof(d_read_ca->head), cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(&(d_read_ca->head), &tmp1, sizeof(tmp1), cudaMemcpyHostToDevice));
    }
}

void write_heads_to_file() {
    FILE *fp;
    fp = fopen("memory_shared_heads.txt", "w");

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            fprintf(fp, "%lf, ", h_ca.head[i * ROWS + j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

int main(void) {
    init_host_ca();

    copy_data_from_CPU_to_GPU();

    perform_simulation_on_GPU();

    copy_data_from_GPU_to_CPU();

    write_heads_to_file();

    return 0;
}

void init_host_ca() {
    h_ca.head = new double[ROWS * COLS]();
    h_ca.Sy = new double[ROWS * COLS]();
    h_ca.K = new double[ROWS * COLS]();
    h_ca.Source = new double[ROWS * COLS]();

    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++) {
            h_ca.head[i * ROWS + j] = headFixed;
            if (j == COLS - 1) {
                h_ca.head[i * ROWS + j] = headCalculated;
            }
            h_ca.Sy[i * ROWS + j] = Syinitial;
            h_ca.K[i * ROWS + j] = Kinitial;
            h_ca.Source[i * ROWS + j] = 0;
        }

    h_ca.Source[posSy * ROWS + posSx] = qw;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line
              << std::endl;
    exit(1);
}

