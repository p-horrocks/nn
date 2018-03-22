#include "layer.h"

Layer::Layer(int numNeurons, int numInputs, bool randomise)
{
    biases_.resize(numNeurons);
    if(randomise)
    {
        std::generate(biases_.begin(), biases_.end(), &normalRand);
    }
    else
    {
        std::fill(biases_.begin(), biases_.end(), (fpt)0);
    }

    weights_.resize(numNeurons, numInputs, randomise);
}

void Layer::setBiases(const fpt_vect& v)
{
    assert(v.size() == biases_.size());
    std::copy(v.begin(), v.end(), biases_.begin());
}

fpt_vect Layer::forward(const fpt_vect& input, fpt_vect* z) const
{
    auto output = weights_.multiply(input);
    ::add(output, biases_);
    if(z)
    {
        z->resize(output.size());
        std::copy(output.begin(), output.end(), z->begin());
    }
    sigmoid(output);
    return output;
}

// Create a layer on the same size but all zero weights and biases
Layer Layer::zeroCopy() const
{
    return Layer(weights_.rows(), weights_.cols(), false);
}

void Layer::add(const Layer& l)
{
    ::add(biases_, l.biases_);
    weights_.add(l.weights_);
}

void Layer::update(const Layer& l, fpt factor)
{
    for(int i = 0; i < biases_.size(); ++i)
    {
        biases_[i] -= factor * l.biases_[i];
    }

    weights_.update(l.weights_, factor);
}
