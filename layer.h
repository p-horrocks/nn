#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

class Layer
{
public:
    Layer(int numNeurons, int numInputs, bool randomise);

    const Matrix& biases() const { return biases_; }
    void setBiases(const Matrix& v);

    const Matrix& weights() const { return weights_; }
    void setWeights(const Matrix& v);

    int size() const   { return weights_.rows(); }
    int inputs() const { return weights_.cols(); }

    void verify(const Matrix& biases, const Matrix& weights) const;

    // Feed the input forward through this layer and return the output. If the
    // z parameter is provided then it will be filled in with the output values
    // before the sigmoid function has been applied.
    //
    // The input and output matrices will be a single column
    Matrix forward(const Matrix& input, Matrix* z = nullptr) const;

    // Create a layer on the same size but all zero weights and biases
    Layer zeroCopy() const;

    // Element-wise addition of both biases and weights. We store the result
    void add(const Layer& l);

    // Element-wise update of biases and weights from training output. Results
    // stored in this. The factor is the learning rate divded by the number of
    // training images that went into the update layer
    void applyUpdate(const Layer& l, fpt decay, fpt factor);

private:
    // The bias matrix will be a single column
    Matrix biases_;
    Matrix weights_;
};

std::ostream& operator << (std::ostream& os, const Layer& l);

#endif // LAYER_H
