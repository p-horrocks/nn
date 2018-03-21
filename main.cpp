#include "network.h"

int main()
{
    Network<float> net({2, 3, 1});

    std::vector<float> in = { 1, 2 };
    auto out = net.forward(in);
    std::cout << out;

    return 0;
}
