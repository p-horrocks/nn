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

fpt_vect Layer::forward(const fpt_vect& input) const
{
    auto output = weights_.multiply(input);
    ::add(output, biases_);
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
