#ifndef MATRIX_H
#define MATRIX_H

#include "funcs.h"

class Matrix
{
public:
    Matrix() {}

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    fpt value(int r, int c) const { return m_[c + (r * cols_)]; }
    void resize(int rows, int cols, bool randomise);

    // Create a single column matrix from the vector
    void fromVector(const fpt_vect& v);

    // Transpose this matrix
    void transpose();

    // Matrix multiplication by a vector
    fpt_vect multiply(const fpt_vect& a) const;

    // Element-wise addition
    void add(const Matrix& a);

    // As per Layer::update
    void update(const Matrix&m, fpt factor);

private:
    int rows_ = 0;
    int cols_ = 0;
    fpt_vect m_;
};

std::ostream& operator << (std::ostream& os, const Matrix& m);

#endif // MATRIX_H
