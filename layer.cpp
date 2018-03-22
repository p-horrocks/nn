#include "layer.h"

Layer::Layer(int numNeurons, int numInputs, bool randomise)
{
    biases_.resize(numNeurons, 1, randomise);
    weights_.resize(numNeurons, numInputs, randomise);
}

void Layer::setBiases(const fpt_vect& v)
{
    assert(0);
//    assert(v.size() == biases_.size());
//    std::copy(v.begin(), v.end(), biases_.begin());
}

Matrix Layer::forward(const Matrix& input, fpt_vect* z) const
{
    auto output = weights_.multiply(input);
    output.add(biases_);
    if(z)
    {
        assert(0);
//        z->resize(output.size());
//        std::copy(output.begin(), output.end(), z->begin());
    }
    output.sigmoid();
    return output;
}

// Create a layer on the same size but all zero weights and biases
Layer Layer::zeroCopy() const
{
    return Layer(weights_.rows(), weights_.cols(), false);
}

void Layer::add(const Layer& l)
{
    biases_.add(l.biases_);
    weights_.add(l.weights_);
}

void Layer::update(const Layer& l, fpt factor)
{
    biases_.update(l.biases_, factor);
    weights_.update(l.weights_, factor);
}
