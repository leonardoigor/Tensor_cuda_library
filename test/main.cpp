#include "iostream"
#include "LibTest.h"
#include "tensor1d.cuh"
int main(int argc, char const *argv[])
{
    LibTest *lt = new LibTest();
    int size1, size2;
    size1 = 20000;
    size2 = 20000;
    Tensor1d *a = new Tensor1d(size1, size2);
    Tensor1d *b = new Tensor1d(size1, size2);
    float *hostData = static_cast<float *>(malloc(sizeof(float) * size1 * size2));
    for (size_t i = 0; i < size1 * size2; i++)
    {
        hostData[i] = (int)i;
    }

    a->setValues(hostData);
    b->setValues(hostData);
    Tensor1d *c = new Tensor1d(size1, size2);
    a->Mul(*b, *c);
    free(hostData);
    // a->print();
    std::cout << lt->sum(2, 3) << std::endl;
    printf("Hello \n");
    // auto c = a * b;
    delete a;
    delete b;

    return 0;
}
