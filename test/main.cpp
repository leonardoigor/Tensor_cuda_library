#include "iostream"
#include "LibTest.h"
#include "tensor.cuh"
int main(int argc, char const *argv[])
{
    LibTest *lt = new LibTest();
    Tensor *tensor = new Tensor(100000, 20000);
    std::cout << lt->sum(2, 3) << std::endl;
    printf("Hello \n");
    delete tensor;

    return 0;
}
