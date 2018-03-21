#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "mnist.h"

class Network
{
public:
    Network(const std::initializer_list<fpt>& init);

    fpt_vect forward(const fpt_vect& input) const;
    void backward(
            const fpt_vect&     input,
            const fpt_vect&     output,
            std::vector<Layer>& delta
            ) const;

    // Train the network using stochastic gradient descent
    void trainMNIST_SGD(
            int epochs,    // number of rounds of training
            int batchSize, // number of images per training round
            fpt rate,      // learning rate
            const Mnist& trainingData,
            const Mnist& testData
            );

    // Train the network on a batch of MNIST images
    void trainMNIST_SGD_batch(
            std::vector<Mnist::ImagePtr>::const_iterator begin,
            std::vector<Mnist::ImagePtr>::const_iterator end,
            fpt rate
            );

    // Returns true if the network correctly outputs the image label
    bool evaluate(const Mnist::ImagePtr& img) const;

    void zeroCopy(std::vector<Layer>& layers) const;

private:
    std::vector<Layer> layers_;
};

#endif // NETWORK_H
