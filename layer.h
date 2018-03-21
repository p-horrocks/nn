#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

class Layer
{
public:
    Layer(int numNeurons, int numInputs, bool randomise);

    fpt_vect forward(const fpt_vect& input) const;

    // Create a layer on the same size but all zero weights and biases
    Layer zeroCopy() const;

    void add(const Layer& l);

private:
    fpt_vect biases_;
    Matrix weights_;
};

#endif // LAYER_H
