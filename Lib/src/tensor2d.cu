#include "include/tensor2d.cuh"
#include "tensor2d.cuh"
#include "tensor2d_kernel.cuh"

Tensor2d::Tensor2d(int cols, int rows) : rows(rows), cols(cols)
{
}

Tensor2d Tensor2d::Random(int min, int max, int cols, int rows)
{
    Tensor2d result(cols, rows);
    float **h_data = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        h_data[i] = (float *)malloc(cols * sizeof(float));
    }
    float **d_data;
    cudaMalloc((void **)&d_data, rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        cudaMalloc((void **)&d_data[i], cols * sizeof(float));
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
    RandomInRangeKernel<<<gridDim, blockDim>>>(min, max, cols, rows, d_data);
    cudaDeviceSynchronize();

    // Copy the randomized data back to host memory
    cudaMemcpy(h_data, d_data, rows * sizeof(float *), cudaMemcpyDeviceToHost);
    for (int i = 0; i < rows; i++)
    {
        cudaMemcpy(h_data[i], d_data[i], cols * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return result;
}

Tensor2d::~Tensor2d()
{
}