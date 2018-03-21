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

    // Element-wise addition of both biases and weights. We store the result
    void add(const Layer& l);

    // Element-wise update of biases and weights from training output. Results
    // stored in this. The factor is the learning rate divded by the number of
    // training images that went into the update layer
    void update(const Layer& l, fpt factor);

private:
    fpt_vect biases_;
    Matrix weights_;
};

#endif // LAYER_H
