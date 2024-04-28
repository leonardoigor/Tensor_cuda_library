#include "include/tensor2d.cuh"
#include "tensor2d.cuh"
#include "tensor2d_kernel.cuh"
#include "iostream"
#include <random>
#include <curand_kernel.h>

#define CUDA_CHECK(call)                                                              \
    do                                                                                \
    {                                                                                 \
        cudaError_t error = call;                                                     \
        if (error != cudaSuccess)                                                     \
        {                                                                             \
            printf("CUDA error at %s:%d code=%d(%s) \"%s\" \n",                       \
                   __FILE__, __LINE__, (int)error, cudaGetErrorString(error), #call); \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

// Function to generate random number between min and max
__device__ double generateRandomNumber(double min, double max, curandState_t *state)
{
    // Generate random number within [min, max] range
    return min + (max - min) * curand_uniform_double(state);
}
__global__ void CreateRandomListMinMax(int min, int max, int cols, int rows, double *data)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    curandState_t state;
    curand_init(clock64(), tid, 0, &state);
    for (int i = tid; i < rows * cols; i += stride)
    {
        auto value = generateRandomNumber(min, max, &state);
        data[i] = value;
    }
}
// Kernel to multiply each element of data with a double value
__global__ void MultiplyData(double multiplier, int cols, int rows, double *data)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < rows * cols; i += stride)
    {
        data[i] *= multiplier;
    }
}
__global__ void SetData(double val, int cols, int rows, double *data)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < rows * cols; i += stride)
    {
        data[i] = val;
    }
}

Tensor2d::Tensor2d(int cols, int rows) : rows(rows), cols(cols)
{
    CUDA_CHECK(cudaMalloc((void **)&data, cols * rows * sizeof(double)));
}

Tensor2d::~Tensor2d()
{
    cudaFree(data);
}
void Tensor2d::print()
{

    double *hostData = new double[rows * cols];
    cudaMemcpy(hostData, data, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "[";
    for (int i = 0; i < rows; ++i)
    {
        std::cout << "[";
        for (int j = 0; j < cols; ++j)
        {
            std::cout << hostData[i * cols + j];
            if (j < cols - 1)
            {
                std::cout << ",";
            }
        }
        std::cout << "]";
        if (i < rows - 1)
        {
            std::cout << ",";
        }
    }
    std::cout << "]" << std::endl;
    delete[] hostData;
}
Tensor2d *Tensor2d::Random(int min, int max, int cols, int rows)
{
    Tensor2d *result = new Tensor2d(cols, rows);
    double *data_d = (double *)malloc(cols * rows * sizeof(double));

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = ((cols * rows) + threadsPerBlock - 1) / threadsPerBlock;

    CreateRandomListMinMax<<<blocksPerGrid, threadsPerBlock>>>(min, max, cols, rows, result->data);

    cudaDeviceSynchronize();
    return result;
}

Tensor2d *Tensor2d::mul(double a)
{
    // Define grid and block dimensions
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = ((cols * rows) + threadsPerBlock - 1) / threadsPerBlock;
    MultiplyData<<<blocksPerGrid, threadsPerBlock>>>(a, cols, rows, data);
    cudaDeviceSynchronize();
    return this;
}
Tensor2d *Tensor2d::setValue(double a)
{
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = ((cols * rows) + threadsPerBlock - 1) / threadsPerBlock;
    SetData<<<blocksPerGrid, threadsPerBlock>>>(a, cols, rows, data);
    cudaDeviceSynchronize();
    return this;
}