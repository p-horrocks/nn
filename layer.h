#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

class Layer
{
public:
    Layer(int numNeurons, int numInputs, bool randomise);

    void setBiases(const fpt_vect& v);

    // Feed the input forward through this layer and return the output. If the
    // z parameter is provided then it will be filled in with the output values
    // before the sigmoid function has been applied.
    //
    // The input and output matrices will be a single column
    Matrix forward(const Matrix& input, fpt_vect* z = nullptr) const;

    // Create a layer on the same size but all zero weights and biases
    Layer zeroCopy() const;

    // Element-wise addition of both biases and weights. We store the result
    void add(const Layer& l);

    // Element-wise update of biases and weights from training output. Results
    // stored in this. The factor is the learning rate divded by the number of
    // training images that went into the update layer
    void update(const Layer& l, fpt factor);

private:
    // The bias matrix will be a single column
    Matrix biases_;
    Matrix weights_;
};

#endif // LAYER_H
