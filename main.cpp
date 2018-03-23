#include <fstream>

#include "mnist.h"
#include "network.h"

int main()
{
    Mnist train;
    int n1 = train.loadTrainingData();

    Mnist eval;
    int n2 = eval.loadEvaluationData();

    Mnist test;
    int n3 = test.loadTestData();

    std::cout << "loaded " << n1 << " training, " << n2 << " eval and " << n3 << " test images" << std::endl;

    //remove-me
    std::ifstream is("/tmp/out.txt");
    Matrix d = pythonRead(is);
    Matrix o = pythonRead(is);
    Matrix b = pythonRead(is);
    Matrix w = pythonRead(is);
    std::cout << w;
    exit(0);

    // Single layer NN to take in the MNIST images and generate 10 outputs -
    // the probability that each image is a corresponding digit
    Network net({784, 30, 10});

    net.trainMNIST_SGD(30, 10, 3.f, train, test);

    return 0;
}
