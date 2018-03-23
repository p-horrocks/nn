#include "network.h"

namespace
{

// Functor object to assist during training
class CostDerivative
{
public:
    CostDerivative(
            const Matrix& a,  // activations
            const Matrix& z,  // z vector
            const fpt_vect& o // expected output
            ) :
        a_(a),
        z_(z),
        o_(o)
    {
    }

    fpt operator () (int r, int c, fpt)
    {
        fpt a = a_.value(r, c);
        fpt z = z_.value(r, c);
        fpt o = o_[r];
        return (a - o) * ::sigmoidPrime(z);
    }

private:
    const Matrix& a_;
    const Matrix& z_;
    const fpt_vect& o_;
};

} // namespace

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

void Network::setLayer(int idx, const Matrix& biases, const Matrix& weights)
{
    assert((idx >= 0 &&) (idx < layers_.size()));
    layers_[idx].setBiases(biases);
    layers_[idx].setWeights(weights);
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
    auto retval = zeroCopy();

    // Feed the input forward through the network, recording the layer
    // activations as we go, as well as the z vectors (the layer output before
    // sigmoid is applied)
    std::vector<Matrix> activations;
    std::vector<Matrix> zVectors;
    const int numLayers = layers_.size();
    activations.resize(numLayers);
    zVectors.resize(numLayers);

    Matrix in;
    in.fromVector(input);

    // Feed-forward pass. This will record the activation and z vectors for
    // all the layers
    for(int l = 0; l < numLayers; ++l)
    {
        const auto& a = (l == 0) ? in : activations[l - 1];
        activations[l] = layers_[l].forward(a, &zVectors[l]);
    }

    //remove-me
//    std::cout << retval.back().size() << " " << retval.back().inputs() << std::endl;
//    exit(0);

    Matrix delta;
    for(int i = numLayers - 1; i >= 0; --i)
    {
        if(i == numLayers - 1)
        {
            delta.resize(activations[i].rows(), 1, false);
            delta.apply(CostDerivative(activations[i], zVectors[i], output));
            //remove-me
            std::cout << delta;
            exit(0);
        }
        else
        {
            auto sp = zVectors[i];
            sp.apply([](int,int,fpt v){ return sigmoidPrime(v); });

            auto l = layers_[i + 1].weights().transpose();
            delta  = l.multiply(delta);
            delta.apply([&](int r, int c, fpt v){ return v * sp.value(r, c); });
        }

        retval[i].setBiases(delta);
        retval[i].setWeights(delta.multiply(activations[i].transpose()));
    }

    return retval;
}

void Network::testBack(
        const Matrix& nabla_b1,
        const Matrix& nabla_w1,
        const Matrix& nabla_b2,
        const Matrix& nabla_w2,
        const Matrix& x,
        const Matrix& y
        ) const
{
    auto retval = zeroCopy();

    // Feed the input forward through the network, recording the layer
    // activations as we go, as well as the z vectors (the layer output before
    // sigmoid is applied)
    std::vector<Matrix> activations;
    std::vector<Matrix> zVectors;
    const int numLayers = layers_.size();
    activations.resize(numLayers);
    zVectors.resize(numLayers);

    Matrix in = x;

    // Feed-forward pass. This will record the activation and z vectors for
    // all the layers
    for(int l = 0; l < numLayers; ++l)
    {
        const auto& a = (l == 0) ? in : activations[l - 1];
        activations[l] = layers_[l].forward(a, &zVectors[l]);
    }

    Matrix delta;
    for(int i = numLayers - 1; i >= 0; --i)
    {
        if(i == numLayers - 1)
        {
            delta.resize(activations[i].rows(), 1, false);
            delta.apply(CostDerivative(activations[i], zVectors[i], y.toVector()));
        }
        else
        {
            auto sp = zVectors[i];
            sp.apply([](int,int,fpt v){ return sigmoidPrime(v); });

            auto l = layers_[i + 1].weights().transpose();
            delta  = l.multiply(delta);
            delta.apply([&](int r, int c, fpt v){ return v * sp.value(r, c); });
        }

        retval[i].setBiases(delta);
        retval[i].setWeights(delta.multiply(activations[i].transpose()));

        //remove-me
        if(i == 0)
        {
            assert(retval[i].biases().isEqual(nabla_b1));
            assert(retval[i].weights().isEqual(nabla_w1));
        }
        else if(i == 1)
        {
            assert(retval[i].biases().isEqual(nabla_b2));
            assert(retval[i].weights().isEqual(nabla_w2));
        }
        else
        {
            assert(!"what?");
        }
    }
    std::cout << "All good" << std::endl;
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
            trainMNIST_SGD_batch(in.begin() + batchStart, in.begin() + batchEnd, rate);
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
