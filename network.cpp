#include "network.h"

Network::Network(const std::initializer_list<fpt>& init)
{
    layers_.reserve(init.end() - init.begin());

    int prevSize = 0;
    for(auto i = init.begin(); i != init.end(); ++i)
    {
        // No layer for the first (input) layer
        if(prevSize > 0)
        {
            layers_.push_back(Layer(*i, prevSize, true));
        }
        prevSize = *i;
    }
}

fpt_vect Network::forward(const fpt_vect& input) const
{
    auto output = input;
    for(auto i = layers_.begin(); i != layers_.end(); ++i)
    {
        output = i->forward(output);
    }
    return output;
}

void Network::backward(
        const fpt_vect&     input,
        const fpt_vect&     output,
        std::vector<Layer>& delta
        ) const
{
}

// Train the network using stochastic gradient descent
void Network::trainMNIST_SGD(
        int epochs,    // number of rounds of training
        int batchSize, // number of images per training round
        fpt rate,      // learning rate
        const Mnist& trainingData,
        const Mnist& testData
        )
{
    for(int e = 0; e < epochs; ++e)
    {
        // Randomise the order of the training images
        auto in = trainingData.images();
        std::random_shuffle(in.begin(), in.end());

        // Update the network using batches of images
        int batchStart = 0;
        while(batchStart < in.size())
        {
            int batchEnd = std::min<int>(in.size(), batchStart + batchSize);
            trainMNIST_SGD_batch(in.begin() + batchStart, in.end() + batchEnd, rate);
            batchStart = batchEnd;
        }

        // Evaluate the network's correctness
        int nGood = 0;
        for(auto i = testData.images().begin(); i != testData.images().end(); ++i)
        {
            if(evaluate(*i))
            {
                ++nGood;
            }
        }
        std::cout << "Epoch[" << e << "] " << nGood << " / " << testData.images().size() << std::endl;
    }
}

// Train the network on a batch of MNIST images
void Network::trainMNIST_SGD_batch(
        std::vector<Mnist::ImagePtr>::const_iterator begin,
        std::vector<Mnist::ImagePtr>::const_iterator end,
        fpt rate
        )
{
    // Create layers for the back-propagation
    std::vector<Layer> layers;
    zeroCopy(layers);

    // In the python version, the (x, y) tuples in mini_batches have x as
    // the image data (a 784 element vector) and y as the expected output
    // (a 10 elementvector with all zeros except a single 1)
    for(auto i = begin; i != end; ++i)
    {
        std::vector<Layer> delta;
        zeroCopy(delta);

        backward((*i)->data, (*i)->label, delta);

        for(int l = 0; l < layers.size(); ++l)
        {
            layers[l].add(delta[l]);
        }
    }

    for(int l = 0; l < layers.size(); ++l)
    {
        layers_[l].update(layers[l], rate / (end - begin));
    }
}

// Returns true if the network correctly outputs the image label
bool Network::evaluate(const Mnist::ImagePtr& img) const
{
    auto output = forward(img->data);
    hardMax(output);
    return std::equal(output.begin(), output.end(), img->label.begin());
}

void Network::zeroCopy(std::vector<Layer>& layers) const
{
    // Create a vector of layers that match ourselves, but with all zero
    // weights and biases
    layers.reserve(layers_.size());
    for(auto i = layers_.begin(); i != layers_.end(); ++i)
    {
        layers.push_back(i->zeroCopy());
    }
}
