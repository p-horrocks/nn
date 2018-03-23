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

    // Single layer NN to take in the MNIST images and generate 10 outputs -
    // the probability that each image is a corresponding digit
    Network net({784, 30, 10});

/*remove-me
    {
        std::ifstream is("/data/nn-test/backward-test.txt");
        Matrix b1 = pythonRead(is);
        Matrix w1 = pythonRead(is);
        Matrix b2 = pythonRead(is);
        Matrix w2 = pythonRead(is);
        Matrix x = pythonRead(is);
        Matrix y = pythonRead(is);
        Matrix nb1 = pythonRead(is);
        Matrix nw1 = pythonRead(is);
        Matrix nb2 = pythonRead(is);
        Matrix nw2 = pythonRead(is);

        net.setLayer(0, b1, w1);
        net.setLayer(1, b2, w2);
        net.verifyLayer(0, b1, w1);
        net.verifyLayer(1, b2, w2);

        auto out = net.backward(x.toVector(), y.toVector());

        assert(out.size() == 2);
        out[1].verify(nb2, nw2);
        out[0].verify(nb1, nw1);

        std::cout << "backward() test successful" << std::endl;
    }
//*/
/*remove-me
    {
        std::ifstream is("/data/nn-test/batch-test2.txt");
        Matrix pre_b1 = pythonRead(is);
        Matrix pre_w1 = pythonRead(is);
        Matrix pre_b2 = pythonRead(is);
        Matrix pre_w2 = pythonRead(is);
        std::vector<Mnist::ImagePtr> images;
        for(int i = 0; i < 10; ++i)
        {
            Matrix x = pythonRead(is);
            Matrix y = pythonRead(is);
            auto image = std::make_shared<Mnist::Image>();
            image->rows  = 28;
            image->cols  = 28;
            image->data  = x.toVector();
            image->label = y.toVector();
            images.push_back(image);
        }
        Matrix nabla_b1 = pythonRead(is);
        Matrix nabla_w1 = pythonRead(is);
        Matrix nabla_b2 = pythonRead(is);
        Matrix nabla_w2 = pythonRead(is);
        Matrix post_b1 = pythonRead(is);
        Matrix post_w1 = pythonRead(is);
        Matrix post_b2 = pythonRead(is);
        Matrix post_w2 = pythonRead(is);

        net.setLayer(0, pre_b1, pre_w1);
        net.setLayer(1, pre_b2, pre_w2);
        net.verifyLayer(0, pre_b1, pre_w1);
        net.verifyLayer(1, pre_b2, pre_w2);

        assert(images.size() == 10);
        auto nabla = net.create_SGD_update(images.begin(), images.end());
        assert(nabla.size() == 2);
        nabla[0].verify(nabla_b1, nabla_w1);
        nabla[1].verify(nabla_b2, nabla_w2);

        net.applyUpdate(nabla, 3.f, images.size());
        net.verifyLayer(0, post_b1, post_w1);
        net.verifyLayer(1, post_b2, post_w2);

        std::cout << "SGD_update() test successful" << std::endl;
    }
//*/
    net.trainMNIST_SGD(30, 10, 3.f, train, test);

    return 0;
}
