#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

template<typename T>
class Layer
{
public:
    Layer(int numNeurons, int numInputs)
    {
        biases_.resize(numNeurons);
        std::generate(biases_.begin(), biases_.end(), &normalRand);

        weights_.resize(numNeurons, numInputs);
    }

    std::vector<T> forward(const std::vector<T>& input) const
    {
        auto output = weights_.multiply(input);
        add(output, biases_);
        sigmoid(output);
        return output;
    }

private:
    std::vector<T> biases_;
    Matrix<T> weights_;
};

#endif // LAYER_H
