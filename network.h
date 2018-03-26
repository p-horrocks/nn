#ifndef NETWORK_H
#define NETWORK_H

#include "cost.h"
#include "layer.h"
#include "mnist.h"

class Network
{
public:
    Network(const std::initializer_list<fpt>& init, const AbstractCostPtr cost);

    const Layer& layer(int idx) const { return layers_[idx]; }
    void setLayer(int idx, const Matrix& biases, const Matrix& weights);
    void verifyLayer(int idx, const Matrix& biases, const Matrix& weights);

    // Evaluates the network output for the given input
    fpt_vect forward(const fpt_vect& input) const;

    // Caculate the gradient of the cost function
    std::vector<Layer> backward(
            const fpt_vect& input,
            const fpt_vect& output
            ) const;

    // Train the network using stochastic gradient descent
    void trainMNIST_SGD(
            int epochs,    // number of rounds of training
            int batchSize, // number of images per training round
            fpt rate,      // learning rate
            fpt lambda,    // regularisation lambda value (0=no regularisation)
            const Mnist& trainingData,
            const Mnist& testData
            );

    // Train the network on a batch of MNIST images
    std::vector<Layer> create_SGD_update(
            std::vector<Mnist::ImagePtr>::const_iterator begin,
            std::vector<Mnist::ImagePtr>::const_iterator end
            ) const;

    void applyUpdate(const std::vector<Layer>& nabla, fpt rate, fpt lambda, fpt batchSize, fpt nImages);

    // Returns true if the network correctly outputs the image label
    bool evaluate(const Mnist::ImagePtr& img) const;

    // Returns a network with the same shape, but all zero weights and biases
    std::vector<Layer> zeroCopy() const;

private:
    std::vector<Layer> layers_;
    AbstractCostPtr cost_;
};

#endif // NETWORK_H
