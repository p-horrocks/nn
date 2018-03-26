#include "network.h"

namespace
{

// Functor object to assist during training
class CostDerivative
{
public:
    CostDerivative(
            const Matrix& a,  // activations
            const Matrix& sp, // sigmoid-prime of z vectors
            const fpt_vect& o // expected output
            ) :
        a_(a),
        sp_(sp),
        o_(o)
    {
    }

    fpt operator () (int r, int c, fpt)
    {
        fpt a  = a_.value(r, c);
        fpt sp = sp_.value(r, c);
        fpt o  = o_[r];
        return (a - o) * sp;
    }

private:
    const Matrix& a_;
    const Matrix& sp_;
    const fpt_vect& o_;
};

} // namespace

Network::Network(const std::initializer_list<fpt>& init, const AbstractCostPtr cost) :
    cost_(cost)
{
    assert(cost_);

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
    assert((idx >= 0) && (idx < layers_.size()));
    layers_[idx].setBiases(biases);
    layers_[idx].setWeights(weights);
}

void Network::verifyLayer(int idx, const Matrix& biases, const Matrix& weights)
{
    assert((idx >= 0) && (idx < layers_.size()));
    layers_[idx].verify(biases, weights);
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

    Matrix delta;
    for(int i = numLayers - 1; i >= 0; --i)
    {
        if(i == numLayers - 1)
        {
            delta = cost_->outputError(zVectors[i], activations[i], output);
        }
        else
        {
            auto sp = zVectors[i];
            sp.apply([](int,int,fpt v){ return sigmoidPrime(v); });

            auto l = layers_[i + 1].weights().transpose();
            delta  = l.multiply(delta);
            delta.apply([&](int r, int c, fpt v){ return v * sp.value(r, c); });
        }

        const auto& a = (i == 0) ? in : activations[i - 1];
        retval[i].setBiases(delta);
        retval[i].setWeights(delta.multiply(a.transpose()));
    }

    return retval;
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

            auto nabla = create_SGD_update(in.begin() + batchStart, in.begin() + batchEnd);
            applyUpdate(nabla, rate, batchSize);

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
std::vector<Layer> Network::create_SGD_update(
        std::vector<Mnist::ImagePtr>::const_iterator begin,
        std::vector<Mnist::ImagePtr>::const_iterator end
        ) const
{
    // Create layers for the back-propagation
    auto nabla = zeroCopy();

    // In the python version, the (x, y) tuples in mini_batches have x as
    // the image data (a 784 element vector) and y as the expected output
    // (a 10 elementvector with all zeros except a single 1)
    for(auto i = begin; i != end; ++i)
    {
        auto delta = backward((*i)->data, (*i)->label);

        for(int l = 0; l < nabla.size(); ++l)
        {
            nabla[l].add(delta[l]);
        }
    }

    return nabla;
}

void Network::applyUpdate(const std::vector<Layer>& nabla, fpt rate, fpt nImages)
{
    const fpt factor = rate / nImages;
    for(int l = 0; l < nabla.size(); ++l)
    {
        layers_[l].applyUpdate(nabla[l], factor);
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
