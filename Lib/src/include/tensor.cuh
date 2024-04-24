#ifndef TENSOR_CUR
#define TENSOR_CUR
#ifdef __CUDACC__ // Check if compiling with NVCC (CUDA compiler)
#define CUDA_CALLABLE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <iostream>

class Tensor1d
{
private:
    int rows;
    int cols;

public:
    float *data;
    // Constructor
    Tensor1d(int rows, int cols);

    // Destructor
    ~Tensor1d();

    // Copy constructor
    Tensor1d(const Tensor1d &other);

    // Assignment operator
    Tensor1d &operator=(const Tensor1d &other);

    // Getters
    int getRows() const;
    int getCols() const;

    // Function to set Tensor1d values
    void setValues(float *hostData);

    // Function to print Tensor1d values
    void print() const;

    // Function to copy Tensor1d data from GPU to CPU
    void toCPU(float *hostData) const;

    // Overloaded operators for element-wise operations

    float &operator()(int i, int j) const;

    void Tensor1d::Mul(Tensor1d *b, Tensor1d *c);
    void Tensor1d::Sum(Tensor1d *b, Tensor1d *c);
    Tensor1d operator+(Tensor1d &other);

    // __device__ Tensor1d operator-(const Tensor1d &other) const;

    // // Matrix multiplication
    Tensor1d operator*(Tensor1d &other);
};

#endif /*TENSOR_CUR*/