#include "include/tensor.cuh"

Tensor::Tensor(int rows, int cols) : rows(rows), cols(cols)
{
    cudaMalloc(&data, rows * cols * sizeof(float));
}

Tensor::~Tensor()
{

    std::cout << "Freeing data" << std::endl;
    cudaFree(data);
}
// Copy constructor
Tensor::Tensor(const Tensor &other) : rows(other.rows), cols(other.cols)
{
    cudaMalloc(&data, rows * cols * sizeof(float));
    cudaMemcpy(data, other.data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
}
// Assignment operator
Tensor &Tensor::operator=(const Tensor &other)
{
    if (this != &other)
    {
        cudaFree(data);
        rows = other.rows;
        cols = other.cols;
        cudaMalloc(&data, rows * cols * sizeof(float));
        cudaMemcpy(data, other.data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    return *this;
}
