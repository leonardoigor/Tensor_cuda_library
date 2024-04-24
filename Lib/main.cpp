#include "tensor.cuh"
#include "iostream"

int main(int argc, char const *argv[])
{
    int size1, size2;
    size1 = 50;
    size2 = 50;
    Tensor1d *a = new Tensor1d(size1, size2);
    Tensor1d *b = new Tensor1d(size1, size2);
    float *hostData = static_cast<float *>(malloc(sizeof(float) * size1 * size2));
    float z = 21;
    for (int i = 0; i < size1 * size2; i++)
    {
        hostData[i] = z;
    }

    a->setValues(hostData);
    b->setValues(hostData);
    // a->print();
    // b->print();
    Tensor1d s = (*a) + (*b);
    auto ss = &s;
    // ss->print();
    Tensor1d *c = new Tensor1d(size1, size2);
    // c->print();
    a->Mul(b, c);
    // c->print();
    auto g = (*a) * (*b);
    for (size_t i = 0; i < 5000; i++)
    {
        (*a) * (*b);
    }

    Tensor1d *gg = &g;
    // gg->print();
    free(hostData);
    delete a;
    delete b;
    delete c;
    std::cout << "Heelo" << std::endl;
    return 0;
}
