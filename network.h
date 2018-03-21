#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "mnist.h"

template<typename T>
class Network
{
public:
    Network(const std::initializer_list<T>& init)
    {
        layers_.reserve(init.end() - init.begin());

        int prevSize = 0;
        for(auto i = init.begin(); i != init.end(); ++i)
        {
            // No layer for the first (input) layer
            if(prevSize > 0)
            {
                layers_.push_back(Layer<T>(*i, prevSize));
            }
            prevSize = *i;
        }
    }

    std::vector<T> forward(const std::vector<T>& input) const
    {
        auto output = input;
        for(auto i = layers_.begin(); i != layers_.end(); ++i)
        {
            output = i->forward(output);
        }
        return output;
    }

    // Train the network using stochastic gradient descent
    void trainMNIST_SGD(
            int epochs,    // number of rounds of training
            int batchSize, // number of images per training round
            T   rate,      // learning rate
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
                //update();
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

    // Returns true if the network correctly outputs the image label
    bool evaluate(const Mnist::ImagePtr& img) const
    {
        std::vector<T> input(img->data.size());
        for(int i = 0; i < img->data.size(); ++i)
        {
            input[i] = static_cast<T>(img->data[i]);
        }

        auto output = forward(input);

        int val = 0;
        T weight = output[0];
        for(int i = 1; i < output.size(); ++i)
        {
            if(output[i] > weight)
            {
                val = i;
                weight = output[i];
            }
        }

        return (val == img->label);
    }

private:
    std::vector<Layer<T>> layers_;
};

#endif // NETWORK_H
