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
    // Generate a random number between 0 and 1
    double randNum = curand_uniform(state);

    // Scale and shift the random number to fit within the desired range
    return min + randNum * (max - min);
}
__global__ void CreateRandomListMinMax(int min, int max, int cols, int rows, double *data)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = 0; i < rows * cols; i++)
    {
        curandState_t state;
        curand_init(0, stride, 0, &state);

        data[i] = generateRandomNumber(min, max, &state);
    }
}

Tensor2d::Tensor2d(int cols, int rows) : rows(rows), cols(cols)
{
}

Tensor2d Tensor2d::Random(int min, int max, int cols, int rows)
{
    Tensor2d result(cols, rows);
    double *data_d = (double *)malloc(cols * rows * sizeof(double));

    CUDA_CHECK(cudaMalloc(&result.data, cols * rows * sizeof(double)));
    int threadsPerBlock = 256;
    int blocksPerGrid = ((cols * rows) + threadsPerBlock - 1) / threadsPerBlock;

    CreateRandomListMinMax<<<blocksPerGrid, threadsPerBlock>>>(min, max, cols, rows, result.data);

    return result;
}

Tensor2d::~Tensor2d()
{
    cudaFree(data);
}
void Tensor2d::Print()
{
}