#include "layer.h"

Layer::Layer(int numNeurons, int numInputs, bool randomise)
{
    biases_.resize(numNeurons, 1, randomise);
    weights_.resize(numNeurons, numInputs, randomise);
}

void Layer::setBiases(const Matrix& v)
{
    assert(v.rows() == biases_.rows());
    assert(v.cols() == biases_.cols());
    biases_ = v;
}

void Layer::setWeights(const Matrix& v)
{
    assert(v.rows() == weights_.rows());
    assert(v.cols() == weights_.cols());
    weights_ = v;
}

void Layer::verify(const Matrix& biases, const Matrix& weights) const
{
    assert(biases_.isEqual(biases));
    assert(weights_.isEqual(weights));
}

Matrix Layer::forward(const Matrix& input, Matrix* z) const
{
    auto output = weights_.multiply(input);
    output.add(biases_);
    if(z)
    {
        *z = output;
    }
    output.apply([](int, int, fpt v){ return ::sigmoid(v); });
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

void Layer::applyUpdate(const Layer& l, fpt decay, fpt factor)
{
    auto b = [&](int r, int c, fpt v){ return (v * decay) - (factor * l.biases_.value(r, c)); };
    auto w = [&](int r, int c, fpt v){ return (v * decay) - (factor * l.weights_.value(r, c)); };

    biases_.apply(b);
    weights_.apply(w);
}

std::ostream& operator << (std::ostream& os, const Layer& l)
{
    const auto& b = l.biases();
    const auto& w = l.weights();
    assert(b.rows() == w.rows());
    for(int j = 0; j < b.rows(); ++j)
    {
        os << "[ " << b.value(j, 0) << " ] [ ";
        for(int i = 0; i < w.cols(); ++i)
        {
            os << w.value(j, i) << " ";
        }
        os << "]" << std::endl;
    }
    return os;
}
