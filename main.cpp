#include "mnist.h"
#include "network.h"

int main()
{
    Network<float> net({2, 3, 1});

    std::vector<float> in = { 1, 2 };
    auto out = net.forward(in);
    std::cout << out;

    Mnist train;
    int n1 = train.loadTrainingData();

    Mnist eval;
    int n2 = eval.loadEvaluationData();

    Mnist test;
    int n3 = test.loadTestData();

    std::cout << "loaded " << n1 << " training, " << n2 << " eval and " << n3 << " test images" << std::endl;
    return 0;
}
