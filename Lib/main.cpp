#include "tensor.cuh"
#include "iostream"

int main(int argc, char const *argv[])
{
    int size1, size2;
    size1 = 100;
    size2 = 100;
    Tensor *a = new Tensor(size1, size2);
    Tensor *b = new Tensor(size1, size2);
    float *hostData = static_cast<float *>(malloc(sizeof(float) * size1 * size2));
    float z = 1;
    for (int i = 0; i < size1 * size2; i++)
    {
        hostData[i] = z + 2;
    }

    a->setValues(hostData);
    b->setValues(hostData);
    Tensor *c = new Tensor(size1, size2);
    c->print();
    a->Mul(b, c);
    c->print();
    auto g = (*a) * (*b);
    for (size_t i = 0; i < 5000; i++)
    {
        (*a) * (*b);
    }

    Tensor *gg = &g;
    gg->print();
    free(hostData);
    delete a;
    delete b;
    delete c;
    std::cout << "Heelo" << std::endl;
    return 0;
}
