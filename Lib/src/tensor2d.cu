#include "include/tensor2d.cuh"
#include "tensor2d.cuh"
#include "tensor2d_kernel.cuh"
#include "iostream"
#include <random>
#include <curand_kernel.h>
#include "utils.cuh"

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