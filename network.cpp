#include "network.h"

#include <list>

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
    Matrix a;
    a.fromVector(input);
    for(auto i = layers_.begin(); i != layers_.end(); ++i)
    {
        a = i->forward(a);
    }
    return a.toVector();
}

std::vector<Layer> Network::backward(
        const fpt_vect& input,
        const fpt_vect& output
        ) const
{
    // Feed the input forward through the network, recording the layer
    // activations as we go, as well as the z vectors (the layer output before
    // sigmoid is applied)
    std::list<fpt_vect> activations;
    std::list<fpt_vect> zVectors;

    // Feed-forward pass
    fpt_vect z;
    auto a = input;
    activations.push_back(a);
    for(auto l = layers_.begin(); l != layers_.end(); ++l)
    {
        //remove-me
//        a = l->forward(a, &z);
//        zVectors.push_back(z);
//        activations.push_back(a);
    }

    auto delta = z;
    //remove-me
//    ::sigmoidPrime(delta);
//    ::dot(delta, costDerivative(a, output));

    // Point to second-last layer
    auto l = activations.rbegin();
    ++l;

    auto retval = zeroCopy();
    retval.back().setBiases(delta);
    //retval.back().setWeights();

    return retval;
/*remove-me
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
*/
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
//remove-me            trainMNIST_SGD_batch(in.begin() + batchStart, in.begin() + batchEnd, rate);
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
    auto layers = zeroCopy();

    // In the python version, the (x, y) tuples in mini_batches have x as
    // the image data (a 784 element vector) and y as the expected output
    // (a 10 elementvector with all zeros except a single 1)
    for(auto i = begin; i != end; ++i)
    {
        auto delta = backward((*i)->data, (*i)->label);

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

std::vector<Layer> Network::zeroCopy() const
{
    // Create a vector of layers that match ourselves, but with all zero
    // weights and biases
    std::vector<Layer> retval;
    retval.reserve(layers_.size());
    for(auto i = layers_.begin(); i != layers_.end(); ++i)
    {
        retval.push_back(i->zeroCopy());
    }

    assert(retval.size() == layers_.size());
    assert(!retval.empty());
    return retval;
}

fpt_vect Network::costDerivative(const fpt_vect& output, const fpt_vect& expected) const
{
    auto retval = output;
    ::sub(retval, expected);
    return retval;
}
